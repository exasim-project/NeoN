# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh, Mesh
from blockamr.field import FaceField


def _make_mesh(ncell=32, max_size=32):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    return Mesh(ba, dm, geom)


def _tag_all(lev, tags, time, ngrow):
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def test_face_field_allocates(blockamr_session):
    """FaceField(mesh, ...) creates face-centred MultiFabs at level 0."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    lev0 = ff[0]
    assert lev0.x.mf is not None
    assert lev0.y.mf is not None
    assert lev0.z.mf is not None
    assert lev0[0] is lev0.x
    assert lev0[1] is lev0.y
    assert lev0[2] is lev0.z


def test_face_field_x_shape(blockamr_session):
    """x-face field has shape (33, 32, 32, 1) for a 32^3 box."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    for mfi in blockamr.MFIterator(ff[0].x.mf):
        arr = ff[0].x.mf.copy_to_host(mfi)
        assert arr.shape[0] == 33
        assert arr.shape[1] == 32
        break


def test_face_field_fill_boundary(blockamr_session):
    """fill_boundary on _FaceFieldLevel fills ghosts on all 3 components."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 5.0
            ff[0][d].mf.copy_from(mfi, arr)
    ff[0].fill_boundary()
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            assert np.allclose(ff[0][d].mf.copy_to_host(mfi), 5.0)


def test_face_field_amr_lifecycle(blockamr_session):
    """FaceField works with AmrMesh: regrid creates fine-level data."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)
    mesh = AmrMesh(geom, info)

    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")
    mesh.init_from_scratch(0.0)
    assert ff[0] is not None
    assert ff[1] is None

    mesh.regrid(0.0, tag=_tag_all)
    assert ff[1] is not None
