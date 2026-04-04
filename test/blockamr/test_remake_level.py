# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Test _on_remake_level data preservation.

Verifies that field data survives regrid via fill_patch:
- No NaN after regrid (set_val(0.0) initializes before fill_patch)
- Data is preserved when box layout doesn't change
- Data is interpolated when box layout changes
"""

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.mesh import AmrMesh
from neon.blockamr.field import CellField
from neon.blockamr.fillpatch import FillPatchCellConservative


def _make_amr_mesh(N=32, Nz=4, max_level=1, max_size=16):
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = max_level
    for lev in range(max_level):
        info.set_ref_ratio(lev, 2)
    info.set_max_grid_size(0, max_size)
    info.set_blocking_factor(0, 4)
    return AmrMesh(geom, info)


def _tag_center(mesh, width=0.2):
    def _tag(lev, tags, time, ngrow):
        dx = mesh.geom(lev).cell_size()
        plo = mesh.geom(lev).prob_lo()
        for tbi in blockamr.TagBoxIterator(tags):
            bx = tbi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nx, ny, nz = hi[0]-lo[0]+1, hi[1]-lo[1]+1, hi[2]-lo[2]+1
            xs = (np.arange(nx) + lo[0] + 0.5) * dx[0] + plo[0]
            ys = (np.arange(ny) + lo[1] + 0.5) * dx[1] + plo[1]
            mask = ((np.abs(xs - 0.5)[:, None] < width)
                    & (np.abs(ys - 0.5)[None, :] < width))
            tbi.set_tags(np.broadcast_to(
                mask[:, :, None], (nx, ny, nz)).astype(np.int32).copy())
    return _tag


def _init_sin_field(phi, mesh):
    """Fill phi with sin(2*pi*x)*sin(2*pi*y) on all levels."""
    for lev in range(mesh.n_levels()):
        if phi.mf[lev] is None:
            continue
        dx = mesh.geom(lev).cell_size()
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            bx = mfi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            nx, ny, nz = hi[0]-lo[0]+1, hi[1]-lo[1]+1, hi[2]-lo[2]+1
            ng = phi.mf[lev].n_grow()
            Nx, Ny, Nz = nx + 2*ng, ny + 2*ng, nz + 2*ng
            xs = (jnp.arange(Nx) + lo[0] - ng + 0.5) * dx[0]
            ys = (jnp.arange(Ny) + lo[1] - ng + 0.5) * dx[1]
            arr = (jnp.sin(2*jnp.pi*xs[:, None, None])
                   * jnp.sin(2*jnp.pi*ys[None, :, None])
                   * jnp.ones((1, 1, Nz)))
            phi.mf[lev].copy_from(mfi, arr[:, :, :, None])
        phi.mf[lev].fill_boundary(mesh.geom(lev))


def _field_stats(phi, mesh):
    """Return (max_abs, has_nan) across all levels."""
    max_abs = 0.0
    has_nan = False
    for lev in range(mesh.n_levels()):
        if phi.mf[lev] is None:
            continue
        for arr in phi.mf[lev].arrays():
            vals = arr[:, :, :, 0]
            max_abs = max(max_abs, float(jnp.max(jnp.abs(vals))))
            if bool(jnp.any(jnp.isnan(vals))):
                has_nan = True
    return max_abs, has_nan


def test_remake_level_no_nan(blockamr_session):
    """After regrid, field data has no NaN (set_val(0.0) before fill_patch)."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                    fill_patch=FillPatchCellConservative())

    tag = _tag_center(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag)
    _init_sin_field(phi, mesh)

    # Regrid — new MultiFabs created via _on_remake_level
    mesh.regrid(0.0, tag=tag)

    _, has_nan = _field_stats(phi, mesh)
    print(f"\nAfter regrid: has_nan={has_nan}")
    assert not has_nan, "Field has NaN after regrid"


def test_remake_level_preserves_data_same_layout(blockamr_session):
    """After regrid with same tagging, field data is preserved."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                    fill_patch=FillPatchCellConservative())

    tag = _tag_center(mesh, width=0.2)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag)
    _init_sin_field(phi, mesh)

    max_before, _ = _field_stats(phi, mesh)
    n_levels_before = mesh.n_levels()
    boxes_before = [len(phi.mf[lev].arrays()) for lev in range(n_levels_before)]

    # Regrid with same tagging → same layout
    mesh.regrid(0.0, tag=tag)

    max_after, has_nan = _field_stats(phi, mesh)
    n_levels_after = mesh.n_levels()
    boxes_after = [len(phi.mf[lev].arrays()) for lev in range(n_levels_after)]

    print(f"\nBefore: max={max_before:.6f}, levels={n_levels_before}, boxes={boxes_before}")
    print(f"After:  max={max_after:.6f}, levels={n_levels_after}, boxes={boxes_after}")

    assert not has_nan, "Field has NaN after same-tag regrid"
    assert n_levels_after == n_levels_before, "Level count changed with same tagging"

    # Data should be preserved — max value should be close
    # (fill_patch_two_levels with cell_cons_interp on same layout = exact copy)
    assert max_after > 0.5 * max_before, (
        f"Field data lost after same-tag regrid: max {max_before:.6f} → {max_after:.6f}")


def test_remake_level_nonzero_after_layout_change(blockamr_session):
    """After regrid that changes layout, field data is interpolated (not zero)."""
    mesh = _make_amr_mesh(N=32, Nz=4, max_level=1)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                    fill_patch=FillPatchCellConservative())

    # Start with small tag
    tag_small = _tag_center(mesh, width=0.1)
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=tag_small)
    _init_sin_field(phi, mesh)

    max_before, _ = _field_stats(phi, mesh)

    # Regrid with wider tag → different box layout on fine level
    tag_wide = _tag_center(mesh, width=0.3)
    mesh.regrid(0.0, tag=tag_wide)

    max_after, has_nan = _field_stats(phi, mesh)

    print(f"\nBefore (small tag): max={max_before:.6f}")
    print(f"After (wide tag):   max={max_after:.6f}, has_nan={has_nan}")

    assert not has_nan, "Field has NaN after layout-change regrid"
    # Data should be interpolated from coarse → fine, not zero
    # The sin wave has max ~1.0, interpolation should preserve most of it
    assert max_after > 0.1, (
        f"Field data near-zero after layout change: max={max_after:.6f}")
