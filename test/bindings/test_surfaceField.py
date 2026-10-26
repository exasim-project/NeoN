# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import neon


def test_scalar_surface_field_and_bcs(executor):
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    bcs = neon.create_calculated_surface_bcs_scalar(mesh)
    assert len(bcs) == mesh.n_boundaries()

    field = neon.ScalarSurfaceField(exec, "phi", mesh)
    assert field.size() == mesh.n_internal_faces()

    neon.fill(field.internal_vector(), 1.0)
    assert neon.equal(field.internal_vector(), 1.0)


def test_vec3_surface_field_arithmetic(executor):
    """scalar * Vec3 (all three spellings), Vec3 + Vec3 and Vec3 - Vec3.

    Only the internal values are checked: VectorSurfaceField exposes no boundary accessor
    to Python. The 1D mesh has internal faces, so the internal-face kernels run.
    """
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)
    n_faces = mesh.n_internal_faces()
    assert n_faces > 0

    s = neon.ScalarSurfaceField(exec, "s", mesh)
    a = neon.VectorSurfaceField(exec, "a", mesh)
    b = neon.VectorSurfaceField(exec, "b", mesh)
    neon.fill(s.internal_vector(), 2.0)
    neon.fill(a.internal_vector(), neon.Vec3(1.0, 2.0, 3.0))
    neon.fill(b.internal_vector(), neon.Vec3(0.5, 0.5, 0.5))

    def values(field):
        return np.asarray(field.internal_vector().copy_to_host())

    def expect(field, vec3):
        assert isinstance(field, neon.VectorSurfaceField)
        np.testing.assert_allclose(values(field), np.tile(vec3, (n_faces, 1)))

    expect(s * a, [2.0, 4.0, 6.0])  # ScalarSurfaceField.__mul__(VectorSurfaceField)
    expect(a * s, [2.0, 4.0, 6.0])  # VectorSurfaceField.__mul__(ScalarSurfaceField)
    expect(a.__rmul__(s), [2.0, 4.0, 6.0])
    expect(a + b, [1.5, 2.5, 3.5])
    expect(a - b, [0.5, 1.5, 2.5])

    # the operands are left untouched
    expect(a, [1.0, 2.0, 3.0])
    expect(b, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(values(s), 2.0)


def test_vec3_surface_field_arithmetic_rejects_mismatched_operands(executor):
    """Operands on different meshes are refused rather than combined element by element."""
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)
    other_mesh = neon.create_1d_uniform_mesh(exec, 6)

    a = neon.VectorSurfaceField(exec, "a", mesh)
    c = neon.VectorSurfaceField(exec, "c", other_mesh)
    s_other = neon.ScalarSurfaceField(exec, "s", other_mesh)

    with pytest.raises(ValueError):
        a + c
    with pytest.raises(ValueError):
        a - c
    with pytest.raises(ValueError):
        s_other * a
