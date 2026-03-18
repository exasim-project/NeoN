# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import numpy as np


def _make_geom():
    n = 64
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    return box, blockamr.Geometry(box, rb, 0, [0, 0, 0])


def test_cell_field_shape():
    """CellField patches are 32x32x32 for max_size=32."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.CellField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for patch in field.patches():
        assert patch.valid_arr.shape == (32, 32, 32, 1)
        break


def test_nodal_field_shape():
    """NodalField patches are 33x33x33 for max_size=32."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.NodalField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for patch in field.patches():
        assert patch.valid_arr.shape == (33, 33, 33, 1)
        break


def test_face_field_x_shape():
    """FaceField.x patches are 33x32x32."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.FaceField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for patch in field.x.patches():
        assert patch.valid_arr.shape == (33, 32, 32, 1)
        break


def test_face_field_y_shape():
    """FaceField.y patches are 32x33x32."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.FaceField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for patch in field.y.patches():
        assert patch.valid_arr.shape == (32, 33, 32, 1)
        break


def test_face_field_z_shape():
    """FaceField.z patches are 32x32x33."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.FaceField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    for patch in field.z.patches():
        assert patch.valid_arr.shape == (32, 32, 33, 1)
        break


def test_face_field_indexing():
    """FaceField[0] is the same as FaceField.x."""
    box, geom = _make_geom()
    dm = blockamr.DistributionMapping(blockamr.BoxArray(box))
    field = blockamr.FaceField(box, dm, geom, ncomp=1, ngrow=0, max_size=32)
    assert field[0] is field.x
    assert field[1] is field.y
    assert field[2] is field.z
