# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

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
