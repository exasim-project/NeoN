# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import neon


def test_scalar_surface_field_division(executor):
    """``/`` evaluates internal and boundary values, for field/field, field/scalar and scalar/field."""
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 8)

    def filled(value_internal, value_boundary):
        field = neon.ScalarSurfaceField(exec, "f", mesh)
        neon.fill(field.internal_vector(), value_internal)
        neon.fill(field.boundary_data_value(), value_boundary)
        return field

    def internal(field):
        return np.asarray(field.internal_vector().copy_to_host())

    def boundary(field):
        return np.asarray(field.boundary_data_value().copy_to_host())

    a = filled(6.0, 60.0)
    b = filled(4.0, 8.0)

    assert np.allclose(internal(a / b), 1.5)
    assert np.allclose(boundary(a / b), 7.5)
    assert np.allclose(internal(a / 4.0), 1.5)
    assert np.allclose(boundary(a / 4.0), 15.0)
    assert np.allclose(internal(12.0 / b), 3.0)
    assert np.allclose(boundary(12.0 / b), 1.5)

    # the operands are left untouched
    assert np.allclose(internal(a), 6.0)
    assert np.allclose(boundary(b), 8.0)


def test_scalar_surface_field_division_rejects_mismatched_meshes(executor):
    name, exec = executor
    mesh_a = neon.create_1d_uniform_mesh(exec, 4)
    mesh_b = neon.create_1d_uniform_mesh(exec, 4)
    a = neon.ScalarSurfaceField(exec, "a", mesh_a)
    b = neon.ScalarSurfaceField(exec, "b", mesh_b)

    with pytest.raises(ValueError):
        a / b
