# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Explicit forward-Euler discretisation of ONE AMR level with AMReX C++ ``ParallelFor`` kernels.

Each spatial term launches its scheme's *accumulate* kernel (``div_*_acc`` /
``laplacian_acc`` / ``grad_acc`` / ``source_acc``), adding ``coeff * op(phi)`` into a
scratch source MultiFab; a generic ``euler_update`` axpy then applies the forward-Euler
step. The term's sign is already folded into ``sp_op.coeff``, so ``coeff`` is passed
verbatim. The caller owns the level loop, ``fill_patch`` and ``average_down``.

One launch per term rather than a fused kernel. A term or scheme with no C++ kernel
(e.g. a callable/variable Laplacian gamma) raises ``NotImplementedError`` naming both.
"""

import blockamr


def euler_step(equation, cell_field, lev, t, dt):
    """In-place forward-Euler update of ``cell_field`` on level ``lev``."""
    ddt_coeff = equation.temporal_ops[0].coeff
    src = _scratch(cell_field, lev)
    _evaluate_spatial_terms(equation.spatial_ops, cell_field, lev, t, src)
    blockamr.euler_update(cell_field.mf[lev], src, dt / ddt_coeff, cell_field.ncomp)


def evaluate(terms, cell_field, lev, t):
    """Per-box source arrays for the spatial ``terms`` on level ``lev``.

    Shape ``(nx, ny, nz)`` for a one-component result, else ``(nx, ny, nz, ncomp)``.
    """
    ncomp = max(getattr(op, "out_ncomp", cell_field.ncomp) for op in terms)
    src = _scratch(cell_field, lev, ncomp)
    _evaluate_spatial_terms(terms, cell_field, lev, t, src)
    # src has ngrow=0, so copy_to_host returns the valid region per box.
    boxes = [src.copy_to_host(mfi) for mfi in blockamr.MFIterator(src)]
    return [b[..., 0] for b in boxes] if ncomp == 1 else boxes


def _scratch(cell_field, lev, ncomp=None):
    """Zeroed scratch source MultiFab with ``ncomp`` components, cached per level.

    Keyed on the per-box valid extents (not just sizes: the kernels index the scratch
    with the field's box indices, so a regrid that moves equal-sized boxes must
    rebuild it), so a regrid changes the key with no separate invalidation channel.
    """
    ncomp = cell_field.ncomp if ncomp is None else ncomp
    mf = cell_field.mf[lev]
    sig = tuple(
        (tuple(mfi.valid_box().small_end()), tuple(mfi.valid_box().big_end()))
        for mfi in blockamr.MFIterator(mf)
    )
    cache = getattr(cell_field, "_cpp_scratch", None)
    if cache is None:
        cache = {}
        cell_field._cpp_scratch = cache
    entry = cache.get(lev)
    if entry is None or entry[0] != sig or entry[1].num_comp() != ncomp:
        mesh = cell_field.mesh
        src = blockamr.MultiFab(
            mesh.box_array(lev),
            mesh.dm(lev),
            ncomp,
            0,
            memory=cell_field._memory,
        )
        entry = (sig, src)
        cache[lev] = entry
    src = entry[1]
    src.set_val(0.0)
    return src


def _evaluate_spatial_terms(spatial_ops, cell_field, lev, t, src):
    """Sum every spatial term into the scratch source: src = Σ coeff·op(phi)."""
    geom = cell_field.mesh.geom(lev)
    for sp_op in spatial_ops:
        build_cpp_kernel = getattr(sp_op, "build_cpp_kernel", None)
        if build_cpp_kernel is None:
            scheme = getattr(sp_op, "scheme", None)
            build_cpp_kernel = getattr(scheme, "build_cpp_kernel", None)
        if build_cpp_kernel is None:
            raise NotImplementedError(
                f"no C++ kernel for term {type(sp_op).__name__!r} "
                f"(scheme {getattr(getattr(sp_op, 'scheme', None), 'type', None)!r})"
            )
        build_cpp_kernel().add_to(src, sp_op, cell_field, lev, geom, t)
