# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from blockamr.mesh import Mesh
from blockamr.field import CellField, FaceField, NodalField


def _make_mesh(n=64, max_size=32):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def test_cell_field_shape():
    """CellField patches are 32x32x32 for max_size=32."""
    mesh = _make_mesh()
    field = CellField(mesh, ncomp=1, ngrow=0)
    for mfi in blockamr.MFIterator(field.mf[0]):
        arr = field.mf[0].copy_to_host(mfi)
        assert arr.shape == (32, 32, 32, 1)
        break


def test_nodal_field_shape():
    """NodalField patches are 33x33x33 for max_size=32."""
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = NodalField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.copy_to_host(mfi)
        assert arr.shape == (33, 33, 33, 1)
        break


def test_face_field_x_shape():
    """FaceField x-component has shape (33, 32, 32, 1)."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0)
    for mfi in blockamr.MFIterator(ff[0].x.mf):
        arr = ff[0].x.mf.copy_to_host(mfi)
        assert arr.shape == (33, 32, 32, 1)
        break


def test_face_field_y_shape():
    """FaceField y-component has shape (32, 33, 32, 1)."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0)
    for mfi in blockamr.MFIterator(ff[0].y.mf):
        arr = ff[0].y.mf.copy_to_host(mfi)
        assert arr.shape == (32, 33, 32, 1)
        break


def test_face_field_z_shape():
    """FaceField z-component has shape (32, 32, 33, 1)."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0)
    for mfi in blockamr.MFIterator(ff[0].z.mf):
        arr = ff[0].z.mf.copy_to_host(mfi)
        assert arr.shape == (32, 32, 33, 1)
        break


def test_face_field_indexing():
    """FaceField[0][d] is the same as FaceField[0].x/y/z."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0)
    assert ff[0][0] is ff[0].x
    assert ff[0][1] is ff[0].y
    assert ff[0][2] is ff[0].z
