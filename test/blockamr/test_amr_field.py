# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh
from blockamr.field import AmrField
from blockamr.fillpatch import FillPatchCellConservative, FillPatchSingleLevel


def _make_geom_and_info(ncell=32, max_level=0):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = max_level
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 32)
    info.set_blocking_factor(0, 8)
    return geom, info


def _tag_all(lev, tags, time, ngrow):
    """Tag every cell for refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        mask = np.ones((nx, ny, nz), dtype=np.int32)
        tbi.set_tags(mask)


def test_amr_field_clear_level():
    """AmrField clears MultiFab when level is cleared."""
    geom, info = _make_geom_and_info(ncell=32, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)

    # Level 0 should be allocated
    assert phi.mf[0] is not None

    # Manually trigger clear_level (simulating what regrid does)
    phi._on_clear_level(0)
    assert phi.mf[0] is None


def test_amr_field_getitem_returns_field():
    """AmrField[lev] returns a single-level Field wrapper."""
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)

    field = phi[0]
    assert field.name == "phi"
    assert field.mf is phi.mf[0]


def test_amr_field_fill_patch_single_level():
    """AmrField.fill_patch on level 0 fills ghost cells."""
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=1)
    mesh.init_from_scratch(0.0)

    # Fill valid + ghost with constant via copy_from (works on device)
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 5.0
        phi.mf[0].copy_from(mfi, arr)

    phi.fill_patch(0, 0.0)

    # After fill_patch, valid cells should still be 5.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr, 5.0)


def test_amr_field_on_new_level():
    """_on_new_level allocates a MultiFab with correct ncomp and ngrow."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=2, ngrow=3)
    mesh.init_from_scratch(0.0)

    assert phi.mf[0].num_comp() == 2
    assert phi.mf[0].n_grow() == 3


def test_amr_field_on_new_level_from_coarse():
    """Regrid that creates level 1 triggers _on_new_level_from_coarse."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)
    assert phi.mf[1] is None

    mesh.regrid(0.0, tag=_tag_all)
    # Level 1 should now be allocated
    assert phi.mf[1] is not None
    assert phi.mf[1].num_comp() == 1


def test_amr_field_fill_patch_strategy_cell_conservative():
    """FillPatchCellConservative fills ghost cells on level 0."""
    geom, info = _make_geom_and_info(ncell=16, max_level=0)
    mesh = AmrMesh(geom, info)
    fp = FillPatchCellConservative()
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=1, fill_patch=fp)
    mesh.init_from_scratch(0.0)

    # Fill valid + ghost with constant
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 3.0
        phi.mf[0].copy_from(mfi, arr)

    phi.fill_patch(0, 0.0)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr, 3.0)


def test_amr_field_fill_patch_strategy_single_level():
    """FillPatchSingleLevel strategy fills ghost cells via FillBoundary."""
    geom, info = _make_geom_and_info(ncell=16, max_level=0)
    mesh = AmrMesh(geom, info)
    fp = FillPatchSingleLevel()
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=1, fill_patch=fp)
    mesh.init_from_scratch(0.0)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 9.0
        phi.mf[0].copy_from(mfi, arr)

    phi.fill_patch(0, 0.0)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr, 9.0)


def test_amr_field_no_fill_patch_uses_fill_boundary():
    """Without a fill_patch strategy, AmrField falls back to fill_boundary."""
    geom, info = _make_geom_and_info(ncell=16, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = AmrField(mesh, name="phi", ncomp=1, ngrow=1)
    mesh.init_from_scratch(0.0)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 11.0
        phi.mf[0].copy_from(mfi, arr)

    phi.fill_patch(0, 0.0)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr, 11.0)
