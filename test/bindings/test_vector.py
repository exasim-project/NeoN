# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

try:
    import jax.numpy as jnp
    has_jax = True
except:
    has_jax = False

import numpy as np
import neon


def test_scalar_vector(executor):
    name, exec = executor

    v1 = neon.ScalarVector(exec, 10)
    assert v1.size() == len(v1) == 10
    assert not v1.empty()

    v2 = neon.ScalarVector(exec, 5, 3.14)
    assert v2.size() == 5

    v3 = neon.ScalarVector(exec, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert v3.size() == 5

    v1.resize(20)
    assert v1.size() == 20

    v_empty = neon.ScalarVector(exec, 0)
    assert v_empty.empty()


def test_vector_vector(executor):
    name, exec = executor

    vv1 = neon.VectorVector(exec, 10)
    assert vv1.size() == 10

    vv2 = neon.VectorVector(exec, 5, neon.Vec3(1.0, 2.0, 3.0))
    assert vv2.size() == 5

    vv3 = neon.VectorVector(exec, [neon.Vec3(i, i+1, i+2) for i in range(3)])
    assert vv3.size() == 3


def test_label_vector(executor):
    name, exec = executor

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


def test_jax_operations(executor):
    """Test JAX operations on NeoN ScalarVectors.

    For CPU executors (Serial/CPU): convert directly via __array__.
    For GPU executors: copy to host first, then convert.
    """
    if not has_jax:
        return
    name, exec = executor

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    v = neon.ScalarVector(exec, values)

    # GPU vectors must be copied to host before converting to JAX
    host_v = v.copy_to_host() if name == "gpu" else v

    arr = jnp.asarray(host_v)
    assert arr.shape == (5,)
    assert float(jnp.sum(arr)) == 15.0
    assert float(jnp.mean(arr)) == 3.0
    assert float(jnp.max(arr)) == 5.0
    assert float(jnp.min(arr)) == 1.0


def test_tensor_vector(executor):
    name, exec = executor

    tv1 = neon.TensorVector(exec, 10)
    assert tv1.size() == len(tv1) == 10

    tv2 = neon.TensorVector(exec, 5, neon.Tensor(2.0))
    assert tv2.size() == 5

    tv3 = neon.TensorVector(
        exec, [neon.Tensor(*[9 * i + k for k in range(9)]) for i in range(3)]
    )
    assert tv3.size() == 3


def test_tensor_vector_numpy_layout():
    """A Vector<Tensor> maps to an (n, 9) row major numpy view."""
    exec = neon.SerialExecutor()

    tensors = [neon.Tensor(*[9 * i + k for k in range(9)]) for i in range(4)]
    tv = neon.TensorVector(exec, tensors)

    arr = np.asarray(tv)
    assert arr.shape == (4, 9)
    assert arr.flags["C_CONTIGUOUS"]
    for i in range(4):
        assert list(arr[i]) == [float(9 * i + k) for k in range(9)]

    # uniform value construction fills every entry with the same tensor
    uniform = np.asarray(neon.TensorVector(exec, 3, neon.Tensor(1.0)))
    assert uniform.shape == (3, 9)
    for i in range(3):
        assert list(uniform[i]) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_tensor_vector_copy_to_host(executor):
    name, exec = executor

    tensors = [neon.Tensor(*[9 * i + k for k in range(9)]) for i in range(3)]
    tv = neon.TensorVector(exec, tensors)

    host = tv.copy_to_host()
    assert host.size() == tv.size() == 3

    arr = np.asarray(host)
    assert arr.shape == (3, 9)
    for i in range(3):
        assert list(arr[i]) == [float(9 * i + k) for k in range(9)]
