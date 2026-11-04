# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_compute_co_num_on_uniform_mesh(executor):
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    face_flux = neon.ScalarSurfaceField(exec, "faceFlux", mesh)
    neon.fill(face_flux.internal_vector(), 1.0)
    neon.fill(face_flux.boundary_data_value(), 1.0)

    max_CoNum, mean_CoNum = neon.compute_co_num(face_flux, 0.01)
    assert abs(max_CoNum - 0.04) < 1.0e-12
    assert abs(mean_CoNum - 0.04) < 1.0e-12
