# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_mesh_imports():
    assert hasattr(neon, 'BoundaryMesh')
    assert hasattr(neon, 'UnstructuredMesh')
    assert hasattr(neon, 'create_single_cell_mesh')
    assert hasattr(neon, 'create_1d_uniform_mesh')


def test_single_cell_mesh(executor):
    name, exec = executor
    mesh = neon.create_single_cell_mesh(exec)

    assert mesh.n_cells() == 1
    assert mesh.n_boundaries() == 4  # left, top, right, bottom
    assert mesh.n_boundary_faces() > 0
    assert mesh.cell_volumes.size() == 1
    assert mesh.cell_centers.size() == 1
    assert mesh.boundary_mesh().face_owners().size() > 0


def test_1d_uniform_mesh(executor):
    name, exec = executor
    n_cells = 10
    mesh = neon.create_1d_uniform_mesh(exec, n_cells, 1.0)

    assert mesh.n_cells() == n_cells
    assert mesh.n_internal_faces() == n_cells - 1
    assert mesh.n_total_faces() > 0
    assert mesh.cell_volumes.size() == n_cells
    assert mesh.cell_centers.size() == n_cells
    assert mesh.face_neighbors.size() == mesh.n_internal_faces()


def test_mesh_geometry(executor):
    name, exec = executor
    mesh = neon.create_single_cell_mesh(exec)

    assert mesh.points.size() > 0
    assert mesh.cell_volumes.size() == mesh.n_cells()
    assert mesh.cell_centers.size() == mesh.n_cells()


def test_mesh_topology(executor):
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 5, 1.0)

    assert mesh.face_neighbors.size() == mesh.n_internal_faces()
    assert mesh.boundary_mesh().face_owners().size() > 0


def test_boundary_mesh_fields(executor):
    name, exec = executor
    mesh = neon.create_single_cell_mesh(exec)
    bm = mesh.boundary_mesh()
    n_bfaces = mesh.n_boundary_faces()

    # check all boundary fields are accessible and sized correctly
    assert bm.face_owners().size() == n_bfaces
    assert bm.face_centers().size() == n_bfaces
    assert bm.face_normals().size() == n_bfaces
    bm.owner_cell_centers()
    bm.face_areas()
    bm.face_unit_normals()
    bm.delta()
    bm.weights()
    bm.delta_coeffs()
    bm.offset()
