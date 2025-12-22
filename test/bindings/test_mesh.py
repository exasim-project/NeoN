# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "neon"))

import neon


def test_mesh_imports():
    assert hasattr(neon, 'BoundaryMesh')
    assert hasattr(neon, 'UnstructuredMesh')
    assert hasattr(neon, 'create_single_cell_mesh')
    assert hasattr(neon, 'create_1d_uniform_mesh')


def test_single_cell_mesh():
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)

    assert mesh.n_cells() == 1
    assert mesh.n_boundaries() == 4  # left, top, right, bottom
    assert mesh.n_boundary_faces() > 0
    assert mesh.cell_volumes().size() == 1
    assert mesh.cell_centres().size() == 1
    assert mesh.boundary_mesh().face_cells().size() > 0
    assert neon.is_serial(mesh.exec())


def test_1d_uniform_mesh():
    exec = neon.SerialExecutor()
    n_cells = 10
    mesh = neon.create_1d_uniform_mesh(exec, n_cells)

    assert mesh.n_cells() == n_cells
    assert mesh.n_internal_faces() == n_cells - 1
    assert mesh.n_faces() > 0
    assert mesh.cell_volumes().size() == n_cells
    assert mesh.cell_centres().size() == n_cells
    assert mesh.face_owner().size() == mesh.n_faces()
    assert mesh.face_neighbour().size() == mesh.n_internal_faces()


def test_mesh_geometry():
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)

    assert mesh.points().size() > 0
    assert mesh.cell_volumes().size() == mesh.n_cells()
    assert mesh.cell_centres().size() == mesh.n_cells()
    assert mesh.face_centres().size() == mesh.n_faces()
    assert mesh.face_areas().size() == mesh.n_faces()
    assert mesh.mag_face_areas().size() == mesh.n_faces()


def test_mesh_topology():
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 5)

    assert mesh.face_owner().size() == mesh.n_faces()
    assert mesh.face_neighbour().size() == mesh.n_internal_faces()
    assert mesh.boundary_mesh().face_cells().size() > 0


def test_boundary_mesh_fields():
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)
    bm = mesh.boundary_mesh()
    n_bfaces = mesh.n_boundary_faces()

    # check all boundary fields are accessible and sized correctly
    assert bm.face_cells().size() == n_bfaces
    assert bm.cf().size() == n_bfaces
    assert bm.sf().size() == n_bfaces
    bm.cn()
    bm.mag_sf()
    bm.nf()
    bm.delta()
    bm.weights()
    bm.delta_coeffs()
    bm.offset()


def test_mesh_with_cpu_executor():
    serial = neon.SerialExecutor()
    mesh_serial = neon.create_1d_uniform_mesh(serial, 5)
    assert neon.is_serial(mesh_serial.exec())
    assert mesh_serial.n_cells() == 5

    # Try CPU executor (may fail if Kokkos threads not initialized)
    try:
        cpu = neon.CPUExecutor()
        mesh_cpu = neon.create_1d_uniform_mesh(cpu, 5)
        assert neon.is_cpu(mesh_cpu.exec())
        assert mesh_serial.n_cells() == mesh_cpu.n_cells()
    except RuntimeError as e:
        if "not initialized" not in str(e):
            raise
