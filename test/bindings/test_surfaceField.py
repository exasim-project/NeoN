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


def test_scalar_surface_field_and_bcs():
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    bcs = neon.create_calculated_surface_bcs_scalar(mesh)
    assert len(bcs) == mesh.n_boundaries()

    field = neon.ScalarSurfaceField(exec, "phi", mesh)
    assert field.size() == mesh.n_internal_faces() + mesh.n_boundary_faces()

    neon.fill(field.internal_vector(), 1.0)
    assert neon.equal(field.internal_vector(), 1.0)
