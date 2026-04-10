# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_scalar_surface_field_and_bcs():
    exec = neon.SerialExecutor()
    n_cells = 4
    mesh = neon.create_1d_uniform_mesh(exec, n_cells, neon.Vec3(0.0), neon.Vec3(1.0))

    bcs = neon.create_calculated_surface_bcs_scalar(mesh)
    assert len(bcs) == mesh.n_boundaries()

    field = neon.ScalarSurfaceField(exec, "phi", mesh)
    assert field.size() == mesh.n_internal_faces() + mesh.n_boundary_faces()

    neon.fill(field.internal_vector(), 1.0)
    assert neon.equal(field.internal_vector(), 1.0)
