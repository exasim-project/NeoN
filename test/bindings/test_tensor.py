# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the NeoN Tensor primitive Python bindings."""

import numpy as np
import pytest

import neon


def test_tensor_construction():
    zero = neon.Tensor()
    assert list(np.asarray(zero).flatten()) == [0.0] * 9

    identity = neon.Tensor(2.0)
    assert list(np.asarray(identity).flatten()) == [
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0,
    ]

    components = [float(k) for k in range(9)]
    from_args = neon.Tensor(*components)
    from_list = neon.Tensor(components)
    assert from_args == from_list
    assert len(from_args) == 9


def test_tensor_construction_wrong_size():
    with pytest.raises(Exception):
        neon.Tensor([1.0, 2.0, 3.0])


def test_tensor_numpy_view():
    t = neon.Tensor(*[float(k) for k in range(9)])
    arr = np.asarray(t)
    assert arr.shape == (3, 3)
    assert arr.flags["C_CONTIGUOUS"]
    assert arr[1, 2] == 5.0


def test_tensor_indexing():
    t = neon.Tensor()
    t[0, 1] = 3.0
    t[2, 0] = -1.0
    assert t[0, 1] == 3.0
    assert t[2, 0] == -1.0
    assert t[1, 1] == 0.0

    with pytest.raises(IndexError):
        t[3, 0]
    with pytest.raises(IndexError):
        t[0, 3] = 1.0


def test_tensor_arithmetic():
    a = neon.Tensor(1.0)
    b = neon.Tensor(2.0)

    assert (a + b) == neon.Tensor(3.0)
    assert (b - a) == neon.Tensor(1.0)
    assert (a * 3.0) == neon.Tensor(3.0)
    assert (3.0 * a) == neon.Tensor(3.0)

    c = neon.Tensor(1.0)
    c += b
    assert c == neon.Tensor(3.0)
    c -= a
    assert c == neon.Tensor(2.0)
    c *= 2.0
    assert c == neon.Tensor(4.0)


def test_tensor_operations():
    t = neon.Tensor(*[float(k) for k in range(9)])

    assert t.trace() == 0.0 + 4.0 + 8.0
    assert np.asarray(t.transpose())[0, 1] == 3.0
    assert t.dot(neon.Vec3(1.0, 0.0, 0.0)) == neon.Vec3(0.0, 3.0, 6.0)

    assert abs(neon.Tensor(1.0).mag() - np.sqrt(3.0)) < 1e-12
    assert abs(neon.mag(neon.Tensor(1.0)) - np.sqrt(3.0)) < 1e-12


def test_tensor_repr():
    t = neon.Tensor()
    assert "Tensor" in repr(t)
    assert str(t).startswith("((")
