# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for creating MultiFab from externally-owned (JAX/numpy) memory."""

import numpy as np
import pytest

import neon.blockamr as blockamr


def _make_mesh(n_cell=32, max_size=16):
    box = blockamr.Box([0, 0, 0], [n_cell - 1] * 3)
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return ba, dm


def test_required_buffer_size():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, ncomp=1, ngrow=0)
    assert size == 32**3  # 8 boxes of 16^3


def test_construct_from_buffer():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 0)
    buf = np.zeros(size)
    mf = blockamr.MultiFab(ba, dm, 1, 0, data=buf)
    assert mf.num_comp() == 1
    assert mf.n_grow() == 0


def test_contiguous_array_matches_source():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 0)
    buf = np.arange(size, dtype=np.float64)
    mf = blockamr.MultiFab(ba, dm, 1, 0, data=buf)
    out = np.asarray(mf.contiguous_array())
    np.testing.assert_array_equal(out, buf)


def test_arrays_return_correct_data():
    ba, dm = _make_mesh(32, 16)
    buf = np.ones(blockamr.MultiFab.required_buffer_size(ba, dm, 1, 0))
    mf = blockamr.MultiFab(ba, dm, 1, 0, data=buf)
    for arr in mf.arrays():
        np.testing.assert_allclose(np.asarray(arr), 1.0)


def test_write_through_buffer():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 0)
    buf = np.zeros(size)
    mf = blockamr.MultiFab(ba, dm, 1, 0, data=buf)
    buf[:] = 42.0
    out = np.asarray(mf.contiguous_array())
    np.testing.assert_array_equal(out, 42.0)


def test_multicomp():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 3, 0)
    buf = np.arange(size, dtype=np.float64)
    mf = blockamr.MultiFab(ba, dm, 3, 0, data=buf)
    assert mf.num_comp() == 3
    assert np.asarray(mf.contiguous_array()).shape[0] == size


def test_with_ghost_cells():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 1)
    buf = np.zeros(size)
    mf = blockamr.MultiFab(ba, dm, 1, 1, data=buf)
    assert mf.n_grow() == 1


def test_fill_boundary():
    ba, dm = _make_mesh(32, 16)
    rb = blockamr.RealBox([0.0] * 3, [1.0] * 3)
    box = blockamr.Box([0, 0, 0], [31] * 3)
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 1)
    buf = np.zeros(size)
    mf = blockamr.MultiFab(ba, dm, 1, 1, data=buf)
    mf.fill_boundary(geom)  # should not crash


def test_wrong_buffer_size_raises():
    ba, dm = _make_mesh(32, 16)
    buf = np.zeros(10)  # too small
    with pytest.raises(RuntimeError):
        blockamr.MultiFab(ba, dm, 1, 0, data=buf)


def test_fab_metadata_consistent():
    ba, dm = _make_mesh(32, 16)
    size = blockamr.MultiFab.required_buffer_size(ba, dm, 1, 0)
    buf = np.zeros(size)
    mf = blockamr.MultiFab(ba, dm, 1, 0, data=buf)
    meta = mf.fab_metadata()
    total = sum(m[1] * m[2] * m[3] * m[4] for m in meta)
    assert total == size
