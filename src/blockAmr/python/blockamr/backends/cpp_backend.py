# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Composable AMReX C++ ``ParallelFor`` explicit backend.

Each spatial term launches its scheme's *accumulate* kernel (``div_*_acc`` /
``laplacian_acc`` / ``grad_acc``), adding ``coeff * op(phi)`` into a scratch source
MultiFab; a generic ``euler_update`` axpy then applies the forward-Euler step. The
term's sign is already folded into ``sp_op.coeff``, so ``coeff`` is passed verbatim.

One launch per term rather than a fused kernel. A term or scheme with no C++ kernel
(including a callable/variable Laplacian gamma) raises ``NotImplementedError`` naming
both — there is deliberately no silent jax fallback.
"""

import blockamr


class CppBackend:
    """Composable AMReX C++ ParallelFor explicit forward-Euler backend."""

    def euler_step(self, equation, cell_field, lev, t, dt):
        ddt_coeff = equation.temporal_ops[0].coeff
        src = self._scratch(cell_field, lev)
        self._evaluate_spatial_terms(equation.spatial_ops, cell_field, lev, t, src)
        blockamr.euler_update(cell_field.mf[lev], src, dt / ddt_coeff, cell_field.ncomp)

    def evaluate(self, terms, cell_field, lev, t):
        src = self._scratch(cell_field, lev)
        self._evaluate_spatial_terms(terms, cell_field, lev, t, src)
        # src has ngrow=0, so copy_to_host returns the valid region per box.
        return [src.copy_to_host(mfi) for mfi in blockamr.MFIterator(src)]

    def _scratch(self, cell_field, lev):
        """Zeroed scratch source MultiFab, cached per (level, box-array generation).

        Keyed on the ``fab_metadata`` box-size signature, so a regrid changes the key
        and rebuilds the scratch with no separate invalidation channel.
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

    def _evaluate_spatial_terms(self, spatial_ops, cell_field, lev, t, src):
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
            build_cpp_kernel().add_to(src, sp_op, cell_field, lev, geom)
