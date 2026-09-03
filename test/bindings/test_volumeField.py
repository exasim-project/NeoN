# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import neon


def test_scalar_volume_field_and_bcs(executor):
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    bcs = neon.create_calculated_volume_bcs_scalar(mesh)
    assert len(bcs) == mesh.n_boundaries()

    field = neon.ScalarVolumeField(exec, "p", mesh)
    assert field.size() == mesh.n_cells()

    neon.fill(field.internal_vector(), 2.0)
    assert neon.equal(field.internal_vector(), 2.0)


def _filled(exec, mesh, value):
    """A scalar volume field of the given uniform value (helper for the op tests)."""
    field = neon.ScalarVolumeField(exec, "f", mesh)
    neon.fill(field.internal_vector(), value)
    return field


def _values(field):
    """The field's internal values as a host NumPy array (works on any executor)."""
    np = pytest.importorskip("numpy")
    return np.asarray(field.internal_vector().copy_to_host())


def test_scalar_field_elementwise_operators(executor):
    """The on-device elementwise operators the turbulence closures need.

    These are elementwise Kokkos kernels, so the assertions hold on every executor
    (serial / cpu / gpu) the fixture parameterises — i.e. the Spalart-Allmaras maths
    (chi/fv1/fv2/Stilda/r/g/fw) can be authored as field ops that run on-device.
    """
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 8)

    a = _filled(exec, mesh, 3.0)
    b = _filled(exec, mesh, 5.0)

    # power: integer, square-root and a fractional exponent (the fw (...)^(1/6) case)
    assert np.allclose(_values(a**3.0), 27.0)
    assert np.allclose(_values(_filled(exec, mesh, 16.0) ** 0.5), 4.0)
    assert np.allclose(_values(_filled(exec, mesh, 64.0) ** (1.0 / 6.0)), 2.0)

    # elementwise max / min against another field and against a scalar
    assert np.allclose(_values(neon.field_max(a, b)), 5.0)
    assert np.allclose(_values(neon.field_min(a, b)), 3.0)
    assert np.allclose(_values(neon.field_max(a, 4.0)), 4.0)
    assert np.allclose(_values(neon.field_min(a, 1.0)), 1.0)

    # reflected scalar-op-field and negation
    assert np.allclose(_values(10.0 - a), 7.0)  # __rsub__
    assert np.allclose(_values(12.0 / a), 4.0)  # __rtruediv__
    assert np.allclose(_values(a - 1.0), 2.0)  # __sub__ (field - scalar)
    assert np.allclose(_values(-a), -3.0)  # __neg__

    # tanh / sqrt (kOmegaSST blending F1/F2 = tanh(...), sqrt(k) arguments)
    assert np.allclose(_values(neon.sqrt(_filled(exec, mesh, 9.0))), 3.0)
    assert np.allclose(_values(neon.tanh(_filled(exec, mesh, 0.0))), 0.0)
    assert np.allclose(_values(neon.tanh(a)), np.tanh(3.0))

    # a small chained expression like a closure would write it stays on-device
    chi = _filled(exec, mesh, 2.0)
    fv1 = chi**3 / (chi**3 + 7.1**3)  # SA fv1(chi)
    assert np.allclose(_values(fv1), 8.0 / (8.0 + 7.1**3))


def test_scalar_field_operators_evaluate_boundaries(executor):
    """Field maths evaluates boundary values too, not just the internal field.

    A closure's ``nut = Cmu k^2/epsilon`` boundary values feed the momentum wall
    fluxes (dev2 stress, nuEff laplacian) — internal-only maths would leave them
    stale at their on-disk values.
    """
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 8)

    def filled(value_internal, value_boundary):
        field = neon.ScalarVolumeField(exec, "f", mesh)
        neon.fill(field.internal_vector(), value_internal)
        neon.fill(field.boundary_data_value(), value_boundary)
        return field

    def boundary(field):
        return np.asarray(field.boundary_data_value().copy_to_host())

    a = filled(3.0, 30.0)
    b = filled(5.0, 50.0)

    assert np.allclose(boundary(a * b), 1500.0)
    assert np.allclose(boundary(a + b), 80.0)
    assert np.allclose(boundary(b / a), 5.0 / 3.0)
    assert np.allclose(boundary(a - 1.0), 29.0)
    assert np.allclose(boundary(2.0 * a), 60.0)
    assert np.allclose(boundary(-a), -30.0)
    assert np.allclose(boundary(a**2.0), 900.0)
    assert np.allclose(boundary(neon.field_max(a, 40.0)), 40.0)
    assert np.allclose(boundary(neon.field_min(a, b)), 30.0)
    assert np.allclose(boundary(neon.sqrt(filled(9.0, 900.0))), 30.0)
    assert np.allclose(boundary(neon.tanh(filled(0.0, 0.0))), 0.0)

    # assign copies boundary values too (mirrors the C++ = operator)
    target = filled(0.0, 0.0)
    target.assign(a * b)
    assert np.allclose(_values(target), 15.0)
    assert np.allclose(boundary(target), 1500.0)


def test_field_field_operators_reject_mismatched_meshes(executor):
    """Field-field maths validates its operands instead of reading out of bounds.

    The kernels iterate over the LEFT operand's size and index the right one directly, so a
    mismatch used to be an out-of-bounds device read rather than a Python-level error.
    """
    name, exec = executor
    mesh_a = neon.create_1d_uniform_mesh(exec, 4)
    mesh_b = neon.create_1d_uniform_mesh(exec, 8)

    a = _filled(exec, mesh_a, 2.0)
    b = _filled(exec, mesh_b, 3.0)

    for op in (
        lambda: a * b,
        lambda: a / b,
        lambda: a + b,
        lambda: a - b,
        lambda: neon.field_max(a, b),
        lambda: neon.field_min(a, b),
        lambda: a.assign(b),
    ):
        with pytest.raises(ValueError):
            op()

    # Same size but a different mesh object is still rejected: element i is a different cell.
    mesh_c = neon.create_1d_uniform_mesh(exec, 4)
    c = _filled(exec, mesh_c, 3.0)
    with pytest.raises(ValueError):
        a * c


def test_surface_field_operators_reject_mismatched_meshes(executor):
    """The scalar SurfaceField binary operators validate their operands too."""
    name, exec = executor
    mesh_a = neon.create_1d_uniform_mesh(exec, 4)
    mesh_b = neon.create_1d_uniform_mesh(exec, 8)

    a = neon.ScalarSurfaceField(exec, "a", mesh_a)
    b = neon.ScalarSurfaceField(exec, "b", mesh_b)
    neon.fill(a.internal_vector(), 2.0)
    neon.fill(b.internal_vector(), 3.0)

    for op in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: a.assign(b)):
        with pytest.raises(ValueError):
            op()

    # The scalar overloads still broadcast.
    assert isinstance(2.0 * a, neon.ScalarSurfaceField)
