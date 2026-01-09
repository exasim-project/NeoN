# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

import neon


def test_surface_interpolation_basic():
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    volume = neon.ScalarVolumeField(exec, "phi", mesh)
    neon.fill(volume.internal_vector(), 1.0)

    interp = neon.SurfaceInterpolationScalar(exec, mesh, "linear")
    surface = interp.interpolate(volume)

    assert surface.size() == mesh.n_internal_faces() + mesh.n_boundary_faces()
