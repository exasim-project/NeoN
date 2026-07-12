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
"""

import neon.blockamr as blockamr

_DIV_ACC = {
    "Upwind": "div_upwind_acc",
    "Linear": "div_linear_acc",
    "VanLeer": "div_vanleer_acc",
    "QUICK": "div_quick_acc",
}


def _scheme_type(sp_op):
    """Resolved scheme identity for error messages / kernel lookup."""
    scheme = getattr(sp_op, "scheme", None)
    if scheme is None:
        return "None"
    return getattr(scheme, "type", type(scheme).__name__)


class CppBackend:
    """Composable AMReX C++ ParallelFor explicit forward-Euler backend."""

    def euler_step(self, equation, cell_field, lev, t, dt):
        ddt_coeff = equation.temporal_ops[0].coeff
        src = self._scratch(cell_field, lev)
        self._accumulate(equation.spatial_ops, cell_field, lev, t, src)
        blockamr.euler_update(cell_field.mf[lev], src, dt / ddt_coeff, cell_field.ncomp)

    def evaluate(self, terms, cell_field, lev, t):
        src = self._scratch(cell_field, lev)
        self._accumulate(terms, cell_field, lev, t, src)
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

    def _accumulate(self, spatial_ops, cell_field, lev, t, src):
        geom = cell_field.mesh.geom(lev)
        phi = cell_field.mf[lev]
        ncomp = cell_field.ncomp
        for sp_op in spatial_ops:
            name = type(sp_op).__name__
            coeff = sp_op.coeff
            if name == "Div":
                stype = _scheme_type(sp_op)
                fn_name = _DIV_ACC.get(stype)
                if fn_name is None:
                    raise NotImplementedError(
                        f"cpp backend: no kernel for term 'Div' with scheme {stype!r}"
                    )
                faces = sp_op.face_field[lev]
                getattr(blockamr, fn_name)(
                    src, phi, faces[0].mf, faces[1].mf, faces[2].mf, geom, coeff, ncomp
                )
            elif name == "Laplacian":
                gamma = sp_op.gamma
                if isinstance(gamma, (int, float)):
                    blockamr.laplacian_acc(src, phi, geom, coeff * float(gamma), ncomp)
                else:
                    raise NotImplementedError(
                        f"cpp backend: variable/callable gamma on term 'Laplacian' "
                        f"(scheme {_scheme_type(sp_op)!r}) not supported (parity with jax)"
                    )
            elif name == "Grad":
                blockamr.grad_acc(src, phi, geom, coeff)
            elif name == "Source":
                raise NotImplementedError(
                    "cpp backend: term 'Source' (spatially-varying coeff_func) not supported"
                )
            else:
                raise NotImplementedError(
                    f"cpp backend: no kernel for term {name!r} (scheme {_scheme_type(sp_op)!r})"
                )
