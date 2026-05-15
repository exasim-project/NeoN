# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_create_uniform_2d_mesh_exists():
    assert hasattr(neon, 'create_uniform_2d_mesh')


def test_create_uniform_2d_mesh_topology():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 3, 2)

    assert mesh.n_cells() == 6
    assert mesh.n_boundaries() == 6  # xmin, xmax, ymin, ymax, zmin, zmax
    assert mesh.n_internal_faces() == 7
    # boundary: xmin(2) + xmax(2) + ymin(3) + ymax(3) + zmin(6) + zmax(6) = 22
    assert mesh.n_boundary_faces() == 22
    assert mesh.n_faces() == 29


def test_create_uniform_2d_mesh_geometry():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 2, 2)

    assert mesh.cell_volumes.size() == 4
    assert mesh.cell_centres.size() == 4
    # Two z-planes: (2+1)*(2+1)*2 = 18 points
    assert mesh.points.size() == 18


def test_create_uniform_2d_mesh_with_domain_size():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 4, 4, 2.0, 2.0)

    assert mesh.n_cells() == 16
    # Two z-planes: (4+1)*(4+1)*2 = 50 points
    assert mesh.points.size() == 50
