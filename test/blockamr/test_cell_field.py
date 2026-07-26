# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh, Mesh
from blockamr.field import CellField, Field


def _make_mesh(ncell=32, max_size=32):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    return Mesh(ba, dm, geom)


def test_cell_field_allocates(blockamr_session):
    """CellField(mesh, ...) allocates MultiFab at level 0."""
    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=2, name="phi")
    assert phi.mf[0] is not None
    assert phi.mf[0].num_comp() == 1
    assert phi.mf[0].n_grow() == 2


def test_cell_field_getitem(blockamr_session):
    """CellField[0] returns a Field view of level 0."""
    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=0, name="phi")
    view = phi[0]
    assert isinstance(view, Field)
    assert view.mf is phi.mf[0]
    assert view.name == "phi"


def test_cell_field_fill_patch_default(blockamr_session):
    """Without fill_patch strategy, fill_patch uses fill_boundary."""
    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 7.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)
    for mfi in blockamr.MFIterator(phi.mf[0]):
        assert np.allclose(phi.mf[0].copy_to_host(mfi), 7.0)


def test_cell_field_clear_level(blockamr_session):
    """_on_clear_level sets mf[lev] to None."""
    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=0, name="phi")
    assert phi.mf[0] is not None
    phi._on_clear_level(0)
    assert phi.mf[0] is None


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


def test_cell_field_with_amr_mesh(blockamr_session):
    """CellField works with AmrMesh: init_from_scratch + regrid."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)
    mesh = AmrMesh(geom, info)

    phi = CellField(mesh, ncomp=1, ngrow=0, name="phi")
    mesh.init_from_scratch(0.0)
    assert phi.mf[0] is not None
    assert phi.mf[1] is None

    mesh.regrid(0.0, tag=_tag_all)
    assert phi.mf[1] is not None
