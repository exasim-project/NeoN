# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

# Add the build directory to path to find the neon module
# The module is built to build/develop/bindings/neon/
build_path = Path(__file__).parent.parent.parent / "build" / "develop" / "bindings" / "neon"
sys.path.insert(0, str(build_path))

import neon


def test_compute_co_num_on_uniform_mesh():
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    face_count = mesh.n_internal_faces() + mesh.n_boundary_faces()
    face_flux = neon.ScalarVector(exec, face_count)
    neon.fill(face_flux, 1.0)

    co_num = neon.compute_co_num(mesh, face_flux, 0.01)
    assert abs(co_num - 0.04) < 1.0e-12
