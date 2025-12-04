# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Comprehensive tests for NeoN mesh Python bindings.

This module tests the Vec3, Vector<T>, BoundaryMesh, and UnstructuredMesh bindings.
"""

import sys
from pathlib import Path

# Add the lib directory to path to find the neon module
sys.path.insert(0, str(Path.cwd() / "neon"))

import neon


def test_imports():
    """Test that all mesh-related classes can be imported."""
    # Vec3
    assert hasattr(neon, 'Vec3')

    # Vector types
    assert hasattr(neon, 'ScalarVector')
    assert hasattr(neon, 'VectorVector')
    assert hasattr(neon, 'LabelVector')

    # Mesh types
    assert hasattr(neon, 'BoundaryMesh')
    assert hasattr(neon, 'UnstructuredMesh')

    # Factory functions
    assert hasattr(neon, 'create_single_cell_mesh')
    assert hasattr(neon, 'create_1d_uniform_mesh')

    print("✓ All mesh-related imports successful")


def test_vec3_creation():
    """Test Vec3 construction and component access."""
    # Default constructor
    v1 = neon.Vec3()
    assert len(v1) == 3
    assert v1[0] == 0.0
    assert v1[1] == 0.0
    assert v1[2] == 0.0

    # Constructor with x, y, z
    v2 = neon.Vec3(1.0, 2.0, 3.0)
    assert v2[0] == 1.0
    assert v2[1] == 2.0
    assert v2[2] == 3.0

    # Constructor with single value
    v3 = neon.Vec3(5.0)
    assert v3[0] == 5.0
    assert v3[1] == 5.0
    assert v3[2] == 5.0

    # Property access
    v4 = neon.Vec3(10.0, 20.0, 30.0)
    assert v4.x == 10.0
    assert v4.y == 20.0
    assert v4.z == 30.0

    print("✓ Vec3 creation tests passed")


def test_vec3_arithmetic():
    """Test Vec3 arithmetic operations."""
    v1 = neon.Vec3(1.0, 2.0, 3.0)
    v2 = neon.Vec3(4.0, 5.0, 6.0)

    # Addition
    v3 = v1 + v2
    assert v3[0] == 5.0
    assert v3[1] == 7.0
    assert v3[2] == 9.0

    # Subtraction
    v4 = v2 - v1
    assert v4[0] == 3.0
    assert v4[1] == 3.0
    assert v4[2] == 3.0

    # Multiplication by scalar
    v5 = v1 * 2.0
    assert v5[0] == 2.0
    assert v5[1] == 4.0
    assert v5[2] == 6.0

    # Reverse multiplication
    v6 = 3.0 * v1
    assert v6[0] == 3.0
    assert v6[1] == 6.0
    assert v6[2] == 9.0

    # In-place operations
    v7 = neon.Vec3(1.0, 1.0, 1.0)
    v7 += neon.Vec3(1.0, 2.0, 3.0)
    assert v7[0] == 2.0
    assert v7[1] == 3.0
    assert v7[2] == 4.0

    print("✓ Vec3 arithmetic tests passed")


def test_vec3_operations():
    """Test Vec3 special operations (dot product, magnitude)."""
    v1 = neon.Vec3(3.0, 4.0, 0.0)

    # Magnitude
    mag = v1.mag()
    assert abs(mag - 5.0) < 1e-10

    # Dot product method
    v2 = neon.Vec3(1.0, 0.0, 0.0)
    dot = v1.dot(v2)
    assert abs(dot - 3.0) < 1e-10

    # Free function magnitude
    mag2 = neon.mag(v1)
    assert abs(mag2 - 5.0) < 1e-10

    # Free function dot product
    v3 = neon.Vec3(1.0, 2.0, 3.0)
    v4 = neon.Vec3(4.0, 5.0, 6.0)
    dot_free = neon.dot(v3, v4)
    expected_dot = 1.0*4.0 + 2.0*5.0 + 3.0*6.0  # = 32.0
    assert abs(dot_free - expected_dot) < 1e-10

    # Verify method and free function give same result
    dot_method = v3.dot(v4)
    assert abs(dot_method - dot_free) < 1e-10

    # Test with orthogonal vectors (dot product should be 0)
    v_x = neon.Vec3(1.0, 0.0, 0.0)
    v_y = neon.Vec3(0.0, 1.0, 0.0)
    dot_orthogonal = neon.dot(v_x, v_y)
    assert abs(dot_orthogonal) < 1e-10

    # Equality
    v5 = neon.Vec3(3.0, 4.0, 0.0)
    assert v1 == v5

    v6 = neon.Vec3(1.0, 2.0, 3.0)
    assert not (v1 == v6)

    print("✓ Vec3 operations tests passed")


def test_vec3_mutability():
    """Test Vec3 component modification."""
    v = neon.Vec3(1.0, 2.0, 3.0)

    # Modify via indexing
    v[0] = 10.0
    assert v[0] == 10.0

    # Modify via properties
    v.y = 20.0
    assert v[1] == 20.0
    assert v.y == 20.0

    v.z = 30.0
    assert v[2] == 30.0

    print("✓ Vec3 mutability tests passed")


def test_scalar_vector():
    """Test ScalarVector creation and basic operations."""
    exec = neon.SerialExecutor()

    # Create uninitialized vector
    v1 = neon.ScalarVector(exec, 10)
    assert v1.size() == 10
    assert len(v1) == 10
    assert not v1.empty()

    # Create with uniform value
    v2 = neon.ScalarVector(exec, 5, 3.14)
    assert v2.size() == 5

    # Create from Python list
    v3 = neon.ScalarVector(exec, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert v3.size() == 5

    # Check executor
    assert neon.is_serial(v1.exec())

    # Resize
    v1.resize(20)
    assert v1.size() == 20

    # Empty check
    v_empty = neon.ScalarVector(exec, 0)
    assert v_empty.empty()
    assert v_empty.size() == 0

    print("✓ ScalarVector tests passed")


def test_vector_vector():
    """Test VectorVector (Vector<Vec3>) operations."""
    exec = neon.SerialExecutor()

    # Create with size
    vv1 = neon.VectorVector(exec, 10)
    assert vv1.size() == 10

    # Create with uniform value
    v_uniform = neon.Vec3(1.0, 2.0, 3.0)
    vv2 = neon.VectorVector(exec, 5, v_uniform)
    assert vv2.size() == 5

    # Create from list
    vec_list = [neon.Vec3(i, i+1, i+2) for i in range(3)]
    vv3 = neon.VectorVector(exec, vec_list)
    assert vv3.size() == 3

    print("✓ VectorVector tests passed")


def test_label_vector():
    """Test LabelVector (Vector<label>) operations."""
    exec = neon.SerialExecutor()

    # Create with size
    lv1 = neon.LabelVector(exec, 10)
    assert lv1.size() == 10

    # Create with uniform value
    lv2 = neon.LabelVector(exec, 5, 42)
    assert lv2.size() == 5

    # Create from list
    lv3 = neon.LabelVector(exec, [0, 1, 2, 3, 4])
    assert lv3.size() == 5

    print("✓ LabelVector tests passed")


def test_single_cell_mesh():
    """Test creation of a single cell mesh."""
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)

    # Check mesh dimensions
    assert mesh.n_cells() == 1
    assert mesh.n_boundaries() == 4  # left, top, right, bottom
    assert mesh.n_boundary_faces() > 0

    # Check that we can access geometric data
    assert mesh.cell_volumes().size() == 1
    assert mesh.cell_centres().size() == 1

    # Check boundary mesh
    bm = mesh.boundary_mesh()
    assert bm.face_cells().size() > 0

    # Check executor
    assert neon.is_serial(mesh.exec())

    print(f"✓ Single cell mesh: {mesh}")


def test_1d_uniform_mesh():
    """Test creation of a 1D uniform mesh."""
    exec = neon.SerialExecutor()
    n_cells = 10
    mesh = neon.create_1d_uniform_mesh(exec, n_cells)

    # Check mesh dimensions
    assert mesh.n_cells() == n_cells
    assert mesh.n_internal_faces() == n_cells - 1  # Each cell connects to next
    assert mesh.n_faces() > 0

    # Check geometric data sizes
    assert mesh.cell_volumes().size() == n_cells
    assert mesh.cell_centres().size() == n_cells

    # Check face connectivity
    # face_owner includes all faces (internal + boundary)
    assert mesh.face_owner().size() == mesh.n_faces()
    # face_neighbour only includes internal faces (boundary faces have no neighbor)
    assert mesh.face_neighbour().size() == mesh.n_internal_faces()

    print(f"✓ 1D uniform mesh with {n_cells} cells: {mesh}")


def test_mesh_geometry_access():
    """Test accessing mesh geometric fields."""
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)

    # Access all geometric fields (should not crash)
    points = mesh.points()
    assert points.size() > 0

    cell_vols = mesh.cell_volumes()
    assert cell_vols.size() == mesh.n_cells()

    cell_centres = mesh.cell_centres()
    assert cell_centres.size() == mesh.n_cells()

    face_centres = mesh.face_centres()
    assert face_centres.size() == mesh.n_faces()

    face_areas = mesh.face_areas()
    assert face_areas.size() == mesh.n_faces()

    mag_face_areas = mesh.mag_face_areas()
    assert mag_face_areas.size() == mesh.n_faces()

    print("✓ Mesh geometry access tests passed")


def test_mesh_topology_access():
    """Test accessing mesh topological data."""
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 5)

    # Face connectivity
    face_owner = mesh.face_owner()
    assert face_owner.size() == mesh.n_faces()

    face_neighbour = mesh.face_neighbour()
    # face_neighbour is only sized for internal faces
    assert face_neighbour.size() == mesh.n_internal_faces()

    # Boundary mesh
    bm = mesh.boundary_mesh()
    face_cells = bm.face_cells()
    assert face_cells.size() > 0

    print("✓ Mesh topology access tests passed")


def test_boundary_mesh_fields():
    """Test accessing BoundaryMesh fields."""
    exec = neon.SerialExecutor()
    mesh = neon.create_single_cell_mesh(exec)
    bm = mesh.boundary_mesh()

    # All these should be accessible without crashing
    face_cells = bm.face_cells()
    cf = bm.cf()
    cn = bm.cn()
    sf = bm.sf()
    mag_sf = bm.mag_sf()
    nf = bm.nf()
    delta = bm.delta()
    weights = bm.weights()
    delta_coeffs = bm.delta_coeffs()
    offset = bm.offset()

    # Check sizes are consistent
    n_boundary_faces = mesh.n_boundary_faces()
    assert face_cells.size() == n_boundary_faces
    assert cf.size() == n_boundary_faces
    assert sf.size() == n_boundary_faces

    print("✓ BoundaryMesh field access tests passed")


def test_copy_to_host():
    """Test copying vectors to host."""
    exec = neon.SerialExecutor()

    # Create a vector and copy to host
    v1 = neon.ScalarVector(exec, [1.0, 2.0, 3.0])
    v2 = v1.copy_to_host()

    # Both should have same size
    assert v1.size() == v2.size()

    # Similar for vector of Vec3
    vv1 = neon.VectorVector(exec, 5)
    vv2 = vv1.copy_to_host()
    assert vv1.size() == vv2.size()

    print("✓ Copy to host tests passed")


def test_mesh_with_different_executors():
    """Test creating meshes with different executors."""
    # Serial executor always works
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
        print("✓ Different executor tests passed (Serial and CPU)")
    except RuntimeError as e:
        if "not initialized" in str(e):
            print("✓ Different executor tests passed (Serial only - CPU requires Kokkos init)")
        else:
            raise


if __name__ == "__main__":
    # Run all tests
    test_imports()
    test_vec3_creation()
    test_vec3_arithmetic()
    test_vec3_operations()
    test_vec3_mutability()
    test_scalar_vector()
    test_vector_vector()
    test_label_vector()
    test_single_cell_mesh()
    test_1d_uniform_mesh()
    test_mesh_geometry_access()
    test_mesh_topology_access()
    test_boundary_mesh_fields()
    test_copy_to_host()
    test_mesh_with_different_executors()

    print("\n" + "="*60)
    print("✓ All mesh binding tests passed successfully!")
    print("="*60)
