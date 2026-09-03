# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import neon


def test_surface_interpolation_scalar(executor):
    name, exec = executor

    mesh = neon.create_1d_uniform_mesh(exec, 4)

    volume = neon.ScalarVolumeField(exec, "phi", mesh)
    neon.fill(volume.internal_vector(), 1.0)

    assert neon.equal(volume.internal_vector(), 1.0)

    token_list = neon.TokenList()
    token_list.insert_string("linear")

    interp = neon.SurfaceInterpolationScalar(exec, mesh, token_list)
    surface = interp.interpolate(volume)

    assert surface.size() == mesh.n_internal_faces()


def test_surface_interpolation_vector(executor):
    name, exec = executor

    mesh = neon.create_1d_uniform_mesh(exec, 4)

    volume = neon.VectorVolumeField(exec, "U", mesh)
    neon.fill(volume.internal_vector(), neon.Vec3(1.0, 2.0, 3.0))

    assert neon.equal(volume.internal_vector(), neon.Vec3(1.0, 2.0, 3.0))

    token_list = neon.TokenList()
    token_list.insert_string("linear")

    interp = neon.SurfaceInterpolationVec3(exec, mesh, token_list)
    surface = interp.interpolate(volume)

    assert surface.size() == mesh.n_internal_faces()


def test_linear_upwind_v_registered_for_vectors_only():
    """linearUpwindV is available to _neon, and only for vector fields.

    A missing scheme aborts the process at factory lookup rather than raising, so this reads the
    registered table instead of trying to construct one and catching.
    """
    schemes = {k: set(v) for k, v in neon.registered_operator_schemes().items()}
    assert "linearUpwindV" in schemes["surfaceInterpolation<Vector>"]
    # Cell limiting is a vector-field concept; the scalar factory must not offer it.
    assert "linearUpwindV" not in schemes["surfaceInterpolation<scalar>"]


def test_surface_interpolation_linear_upwind_v(executor):
    """The linearUpwindV scheme constructs and interpolates through the bindings."""
    name, exec = executor

    mesh = neon.create_1d_uniform_mesh(exec, 4)

    volume = neon.VectorVolumeField(exec, "U", mesh)
    neon.fill(volume.internal_vector(), neon.Vec3(1.0, 2.0, 3.0))
    volume.correct_boundary_conditions()

    flux = neon.ScalarSurfaceField(exec, "flux", mesh)
    neon.fill(flux.internal_vector(), 1.0)

    interp = neon.SurfaceInterpolationVec3(exec, mesh, neon.TokenList(["linearUpwindV", "Gauss"]))
    surface = neon.VectorSurfaceField(exec, "out", mesh)
    interp.interpolate_into(flux, volume, surface)

    # A uniform field has zero gradient, so the (limited) correction vanishes and every face
    # takes the upwind cell value.
    np = pytest.importorskip("numpy")
    values = np.asarray(surface.internal_vector().copy_to_host())
    assert np.allclose(values[: mesh.n_internal_faces()], [1.0, 2.0, 3.0])
