# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Composable AMReX C++ ``ParallelFor`` explicit backend (plan 03 §3).

Each spatial term is discretised by launching the matching *accumulate*
kernel (``div_*_acc`` / ``laplacian_acc`` / ``grad_acc``) which adds
``coeff * op(phi)`` into a scratch source MultiFab; one generic
``euler_update`` axpy then applies the forward-Euler step. The term's sign
is already folded into ``sp_op.coeff`` (see ``FusedEulerKernel``:
``phi - dt_over_coeff * sum(coeff_i * op_i(phi))``), so the accumulate call
passes ``coeff`` verbatim.

One kernel launch per term (correctness-first, not fused) — the fused
``euler_step_*`` kernels stay as perf baselines. No jax import on the hot
path: a term/scheme without a C++ kernel (or a callable/variable Laplacian
gamma) raises ``NotImplementedError`` naming the term and scheme, never a
silent jax fallback.

Dispatch mirrors the jax backend: each term's cpp kernel is owned by its
**scheme** via ``build_cpp_kernel()`` (peer of ``build_spatial_kernel()``),
returning a wrapper from :mod:`~blockamr.cpp_kernels`. This backend just
asks every op's scheme for its kernel and applies it — no per-term ``if/elif``,
no scheme-name-to-binding table.
"""

import blockamr


class CppBackend:
    """Composable AMReX C++ ParallelFor explicit forward-Euler backend."""

    def euler_step(self, equation, cell_field, lev, t, dt):
        ddt_coeff = equation.temporal_ops[0].coeff
        src = self.source(equation.spatial_ops, cell_field, lev, t)
        blockamr.euler_update(cell_field.mf[lev], src, dt / ddt_coeff, cell_field.ncomp)

    def source(self, terms, cell_field, lev, t, ibm=None):
        """Accumulated source (scratch, ngrow=0): ``Σ coeff·op(phi)``, with the
        wall sweep applied when ``ibm`` is given. The R4 seam: operator
        evaluation and time update stay separate named launches. The returned
        scratch is reused (and zeroed) by the next ``source`` call — consume
        it first.
        """
        src = self._scratch(cell_field, lev)
        # W1 (design §5): a wide interior scheme degrades against the marker,
        # in its own kernel and per cell. `None` on every path but an IBM one
        # with a compiled pair, so the plain sweep is the plain call.
        cell_type = None if ibm is None else ibm.interior_cell_type(lev)
        self._evaluate_spatial_terms(terms, cell_field, lev, t, src, cell_type)
        if ibm is not None:
            # The wall sweep, after every term's interior sweep and never
            # fused into one (row-format rule R4). `ibm is None` is this one
            # branch, outside the kernel — which is what makes the no-IBM
            # result bitwise the plain operator's.
            ibm.apply(src, lev, t)
        return src

    def evaluate(self, terms, cell_field, lev, t, ibm=None):
        src = self.source(terms, cell_field, lev, t, ibm=ibm)
        # src has ngrow=0 → copy_to_host returns the valid region per box.
        return [src.copy_to_host(mfi) for mfi in blockamr.MFIterator(src)]

    # -- internals ----------------------------------------------------------

    def _scratch(self, cell_field, lev):
        """Zeroed scratch source MultiFab, cached per (level, box-array gen).

        Keyed on the ``fab_metadata`` box-size signature — the same
        box-array-generation proxy ``ImplicitSolveCache`` compares. A regrid
        rebuilds ``cell_field.mf[lev]`` on a new box array, the signature
        changes, and the scratch is rebuilt (no separate invalidation channel).
        """
        mf = cell_field.mf[lev]
        sig = tuple((m[1], m[2], m[3]) for m in mf.fab_metadata())
        cache = getattr(cell_field, "_cpp_scratch", None)
        if cache is None:
            cache = {}
            cell_field._cpp_scratch = cache
        entry = cache.get(lev)
        if entry is None or entry[0] != sig:
            mesh = cell_field.mesh
            src = blockamr.MultiFab(
                mesh.box_array(lev),
                mesh.dm(lev),
                cell_field.ncomp,
                0,
                memory=cell_field._memory,
            )
            entry = (sig, src)
            cache[lev] = entry
        src = entry[1]
        src.set_val(0.0)
        return src

    def _evaluate_spatial_terms(self, spatial_ops, cell_field, lev, t, src, cell_type=None):
        """Sum every spatial term into the scratch source: src = Σ coeff·op(phi)."""
        geom = cell_field.mesh.geom(lev)
        for sp_op in spatial_ops:
            scheme = getattr(sp_op, "scheme", None)
            build_cpp_kernel = getattr(scheme, "build_cpp_kernel", None)
            if build_cpp_kernel is None:
                raise NotImplementedError(
                    f"cpp backend: no kernel for term {type(sp_op).__name__!r} "
                    f"(scheme {getattr(scheme, 'type', None)!r})"
                )
            build_cpp_kernel().add_to(src, sp_op, cell_field, lev, geom, cell_type)
