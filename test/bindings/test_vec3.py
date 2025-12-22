# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "neon"))

import pytest
import neon


def test_vec3_creation():
    v1 = neon.Vec3()
    assert len(v1) == 3
    assert v1[0] == v1[1] == v1[2] == 0.0

    v2 = neon.Vec3(1.0, 2.0, 3.0)
    assert (v2[0], v2[1], v2[2]) == (1.0, 2.0, 3.0)

    v3 = neon.Vec3(5.0)
    assert v3[0] == v3[1] == v3[2] == 5.0

    v4 = neon.Vec3(10.0, 20.0, 30.0)
    assert (v4.x, v4.y, v4.z) == (10.0, 20.0, 30.0)


def test_vec3_arithmetic():
    v1 = neon.Vec3(1.0, 2.0, 3.0)
    v2 = neon.Vec3(4.0, 5.0, 6.0)

    v3 = v1 + v2
    assert (v3[0], v3[1], v3[2]) == (5.0, 7.0, 9.0)

    v4 = v2 - v1
    assert (v4[0], v4[1], v4[2]) == (3.0, 3.0, 3.0)

    v5 = v1 * 2.0
    assert (v5[0], v5[1], v5[2]) == (2.0, 4.0, 6.0)

    v6 = 3.0 * v1
    assert (v6[0], v6[1], v6[2]) == (3.0, 6.0, 9.0)

    v7 = neon.Vec3(1.0, 1.0, 1.0)
    v7 += neon.Vec3(1.0, 2.0, 3.0)
    assert (v7[0], v7[1], v7[2]) == (2.0, 3.0, 4.0)


def test_vec3_dot_and_magnitude():
    v1 = neon.Vec3(3.0, 4.0, 0.0)  # classic 3-4-5 triangle
    assert v1.mag() == pytest.approx(5.0)
    assert neon.mag(v1) == pytest.approx(5.0)

    v2 = neon.Vec3(1.0, 0.0, 0.0)
    assert v1.dot(v2) == pytest.approx(3.0)

    v3 = neon.Vec3(1.0, 2.0, 3.0)
    v4 = neon.Vec3(4.0, 5.0, 6.0)
    assert neon.dot(v3, v4) == pytest.approx(32.0)  # 1*4 + 2*5 + 3*6
    assert v3.dot(v4) == pytest.approx(neon.dot(v3, v4))

    # orthogonal vectors
    assert neon.dot(neon.Vec3(1, 0, 0), neon.Vec3(0, 1, 0)) == pytest.approx(0.0)


def test_vec3_equality():
    v1 = neon.Vec3(3.0, 4.0, 0.0)
    v2 = neon.Vec3(3.0, 4.0, 0.0)
    v3 = neon.Vec3(1.0, 2.0, 3.0)

    assert v1 == v2
    assert not (v1 == v3)


def test_vec3_mutability():
    v = neon.Vec3(1.0, 2.0, 3.0)
    v[0] = 10.0
    assert v[0] == 10.0

    v.y = 20.0
    assert v[1] == 20.0

    v.z = 30.0
    assert v[2] == 30.0
    assert v.y == 20.0

    v.z = 30.0
    assert v[2] == 30.0
