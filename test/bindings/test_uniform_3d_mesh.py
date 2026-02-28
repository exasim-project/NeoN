# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_create_uniform_3d_mesh_exists():
    assert hasattr(neon, 'create_uniform_3d_mesh')


def test_create_uniform_3d_mesh_topology():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_3d_mesh(exec, 3, 2, 2)

    assert mesh.n_cells() == 12
    assert mesh.n_boundaries() == 6
    assert mesh.n_internal_faces() == 20
    # boundary: left(4) + right(4) + bottom(6) + top(6) + front(6) + back(6) = 32
    assert mesh.n_boundary_faces() == 32
    assert mesh.n_faces() == 52


def test_create_uniform_3d_mesh_geometry():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_3d_mesh(exec, 2, 2, 2)

    assert mesh.cell_volumes.size() == 8
    assert mesh.cell_centres.size() == 8
    # (2+1)*(2+1)*(2+1) = 27 points
    assert mesh.points.size() == 27


def test_create_uniform_3d_mesh_with_domain_size():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_3d_mesh(exec, 4, 4, 4, 2.0, 2.0, 2.0)

    assert mesh.n_cells() == 64
    # (4+1)*(4+1)*(4+1) = 125 points
    assert mesh.points.size() == 125
