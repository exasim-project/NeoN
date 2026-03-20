# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
import numpy as np
from blockamr.field import Field


def _make_periodic_setup(n_cell=64, max_size=32, ngrow=1):
    """Create a periodic domain with ghost cells."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return mf, geom


def _make_field(n_cell=64, max_size=32, ngrow=1, name="phi"):
    """Create a periodic Field wrapping a MultiFab + Geometry."""
    mf, geom = _make_periodic_setup(n_cell, max_size, ngrow)
    return Field(mf, geom, name=name)


def test_field_patches():
    """Field.patches() yields PatchData with valid and grown arrays."""
    field = _make_field(n_cell=64, max_size=32, ngrow=1)
    count = 0
    for patch in field.patches():
        assert patch.valid_arr.shape[:3] == (32, 32, 32)
        assert patch.grown_arr.shape[:3] == (34, 34, 34)
        assert patch.ngrow == 1
        count += 1
    assert count == 8  # 2x2x2 patches


def test_field_fill_boundary():
    """Field.fill_boundary() delegates to MultiFab.fill_boundary(geom)."""
    field = _make_field(n_cell=64, max_size=64, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        arr[:, :, :, 0] = 42.0

    field.fill_boundary()

    for patch in field.patches():
        ng = patch.ngrow
        assert np.allclose(patch.grown_arr[0, ng, ng, 0], 42.0)


def test_n_grow():
    """MultiFab.n_grow() returns the number of ghost cells."""
    mf, _ = _make_periodic_setup(ngrow=2)
    assert mf.n_grow() == 2


def test_grown_array_shape():
    """grown_array() returns an array that includes ghost cells on each side."""
    n_cell = 64
    max_size = 32
    ngrow = 1
    mf, geom = _make_periodic_setup(n_cell, max_size, ngrow)

    for mfi in blockamr.MFIterator(mf):
        grown = mf.host_grown_array(mfi)
        valid = mf.host_array(mfi)

        assert valid.shape[0] == max_size
        assert valid.shape[1] == max_size
        assert valid.shape[2] == max_size

        assert grown.shape[0] == max_size + 2 * ngrow
        assert grown.shape[1] == max_size + 2 * ngrow
        assert grown.shape[2] == max_size + 2 * ngrow
        break


def test_array_fortran_order():
    """array() returns data with Fortran-order strides (x varies fastest)."""
    mf, geom = _make_periodic_setup(n_cell=64, max_size=32, ngrow=0)

    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_array(mfi)
        assert arr.strides[0] < arr.strides[1]
        assert arr.strides[1] < arr.strides[2]
        break


def test_fill_boundary_periodic():
    """fill_boundary() copies valid data into ghost cells for periodic BCs."""
    n_cell = 64
    max_size = 64
    ngrow = 1
    mf, geom = _make_periodic_setup(n_cell, max_size, ngrow)

    for mfi in blockamr.MFIterator(mf):
        arr = mf.host_array(mfi)
        nx = arr.shape[0]
        for i in range(nx):
            arr[i, :, :, 0] = float(i)

    mf.fill_boundary(geom)

    for mfi in blockamr.MFIterator(mf):
        grown = mf.host_grown_array(mfi)
        ng = ngrow

        assert np.allclose(grown[0, ng, ng, 0], n_cell - 1), (
            f"Left ghost expected {n_cell-1}, got {grown[0, ng, ng, 0]}"
        )
        assert np.allclose(grown[-1, ng, ng, 0], 0.0), (
            f"Right ghost expected 0, got {grown[-1, ng, ng, 0]}"
        )
        break


def test_geometry_prob_lo_hi():
    """Geometry exposes prob_lo() and prob_hi()."""
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    lo = geom.prob_lo()
    hi = geom.prob_hi()
    assert lo == [0.0, 0.0, 0.0]
    assert hi == [1.0, 2.0, 3.0]
