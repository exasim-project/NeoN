# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "neon"))

import neon


def test_scalar_vector():
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 10)
    assert v1.size() == len(v1) == 10
    assert not v1.empty()
    assert neon.is_serial(v1.exec())

    v2 = neon.ScalarVector(exec, 5, 3.14)
    assert v2.size() == 5

    v3 = neon.ScalarVector(exec, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert v3.size() == 5

    v1.resize(20)
    assert v1.size() == 20

    v_empty = neon.ScalarVector(exec, 0)
    assert v_empty.empty()


def test_vector_vector():
    exec = neon.SerialExecutor()

    vv1 = neon.VectorVector(exec, 10)
    assert vv1.size() == 10

    vv2 = neon.VectorVector(exec, 5, neon.Vec3(1.0, 2.0, 3.0))
    assert vv2.size() == 5

    vv3 = neon.VectorVector(exec, [neon.Vec3(i, i+1, i+2) for i in range(3)])
    assert vv3.size() == 3


def test_label_vector():
    exec = neon.SerialExecutor()

    lv1 = neon.LabelVector(exec, 10)
    assert lv1.size() == 10

    lv2 = neon.LabelVector(exec, 5, 42)
    assert lv2.size() == 5

    lv3 = neon.LabelVector(exec, [0, 1, 2, 3, 4])
    assert lv3.size() == 5


def test_copy_to_host():
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, [1.0, 2.0, 3.0])
    v2 = v1.copy_to_host()
    assert v1.size() == v2.size()

    vv1 = neon.VectorVector(exec, 5)
    vv2 = vv1.copy_to_host()
    assert vv1.size() == vv2.size()
