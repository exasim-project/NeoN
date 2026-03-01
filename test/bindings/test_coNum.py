# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_compute_co_num_on_uniform_mesh():
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    n_internal = mesh.n_internal_faces()
    n_boundary = mesh.n_boundary_faces()
    # Only x-direction faces carry flux; y/z boundary faces are zero
    # Internal faces (all x-normal) + xmin(1) + xmax(1) = n_internal + 2
    values = [1.0] * n_internal + [1.0, 1.0] + [0.0] * (n_boundary - 2)
    face_flux = neon.ScalarVector(exec, values)

    max_CoNum, mean_CoNum = neon.compute_co_num(mesh, face_flux, 0.01)
    assert abs(max_CoNum - 0.04) < 1.0e-12
    assert abs(mean_CoNum - 0.04) < 1.0e-12
