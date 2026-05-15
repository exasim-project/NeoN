# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests to verify reference CGNS mesh integrity using h5py and pyvista.

These tests validate that the gmsh-generated reference meshes have the
expected structure and properties, ensuring the C++ reader has valid
inputs to work with.
"""

import h5py
import numpy as np
import pytest


class TestSingleTetMesh:
    """Verify the single tetrahedron reference mesh."""

    def test_has_one_volume_element(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        assert tets.n_cells == 1

    def test_has_four_points(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        assert tets.n_points == 4

    def test_volume_is_one_sixth(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(volumes[0], 1.0 / 6.0, rtol=1e-10)

    def test_points_at_expected_coordinates(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        pts = tets.points
        expected = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(
            np.sort(pts, axis=0), np.sort(expected, axis=0), atol=1e-12
        )


class TestCube3DMesh:
    """Verify the 3D cube reference mesh."""

    def test_has_cells(self, cube3d_grid):
        assert cube3d_grid.n_cells > 0

    def test_total_volume_is_one(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(np.sum(volumes), 1.0, rtol=1e-10)

    def test_all_volumes_positive(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        assert np.all(volumes > 0)

    def test_points_within_unit_cube(self, cube3d_grid):
        pts = cube3d_grid.points
        np.testing.assert_array_less(-1e-12, pts)
        np.testing.assert_array_less(pts, 1.0 + 1e-12)

    def test_surface_area_is_six(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        surface = tets.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 6.0, rtol=1e-6)

    def test_has_six_boundary_patches(self, cube3d_path):
        """Verify 6 named boundary element sections exist in the CGNS file."""
        with h5py.File(str(cube3d_path), "r") as f:
            base_name = [
                k for k in f.keys()
                if k != "CGNSLibraryVersion" and isinstance(f[k], h5py.Group)
            ][0]
            base = f[base_name]
            zone_name = [
                k for k in base.keys()
                if isinstance(base[k], h5py.Group)
                and base[k].attrs.get("label", b"") == b"Zone_t"
            ][0]
            zone = base[zone_name]

            boundary_sections = []
            for name in zone:
                node = zone[name]
                if node.attrs.get("label", b"") == b"Elements_t":
                    elem_type = node[" data"][...].flatten()[0]
                    if elem_type == 5:  # TRI_3
                        boundary_sections.append(name)

            assert len(boundary_sections) == 6


class TestCavity2DMesh:
    """Verify the 2D cavity reference mesh."""

    def test_has_cells(self, cavity2d_grid):
        assert cavity2d_grid.n_cells > 0

    def test_points_within_unit_square(self, cavity2d_grid):
        pts = cavity2d_grid.points
        np.testing.assert_array_less(-1e-12, pts[:, :2])
        np.testing.assert_array_less(pts[:, :2], 1.0 + 1e-12)

    def test_total_area_is_one(self, cavity2d_grid):
        tris = cavity2d_grid.extract_cells_by_type(5)  # VTK_TRIANGLE
        areas = tris.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 1.0, rtol=1e-10)

    def test_has_four_boundary_patches(self, cavity2d_path):
        """Verify 4 named boundary patches (top, bottom, left, right)."""
        with h5py.File(str(cavity2d_path), "r") as f:
            base_name = [
                k for k in f.keys()
                if k != "CGNSLibraryVersion" and isinstance(f[k], h5py.Group)
            ][0]
            base = f[base_name]
            zone_name = [
                k for k in base.keys()
                if isinstance(base[k], h5py.Group)
                and base[k].attrs.get("label", b"") == b"Zone_t"
            ][0]
            zone = base[zone_name]

            boundary_sections = []
            for name in zone:
                node = zone[name]
                if node.attrs.get("label", b"") == b"Elements_t":
                    elem_type = node[" data"][...].flatten()[0]
                    if elem_type == 3:  # BAR_2
                        boundary_sections.append(name)

            assert len(boundary_sections) == 4


def _get_cgns_zone(f):
    """Navigate CGNS HDF5 file to get the first zone."""
    base_name = [
        k for k in f.keys()
        if k != "CGNSLibraryVersion" and isinstance(f[k], h5py.Group)
    ][0]
    base = f[base_name]
    zone_name = [
        k for k in base.keys()
        if isinstance(base[k], h5py.Group)
        and base[k].attrs.get("label", b"") == b"Zone_t"
    ][0]
    return base[zone_name]


class TestCGNSFileStructure:
    """Verify CGNS file structure is valid for NeoN reader."""

    def test_cube3d_has_zone_bc(self, cube3d_path):
        with h5py.File(str(cube3d_path), "r") as f:
            zone = _get_cgns_zone(f)
            assert "ZoneBC" in zone

    def test_cube3d_has_at_least_six_bcs(self, cube3d_path):
        with h5py.File(str(cube3d_path), "r") as f:
            zone = _get_cgns_zone(f)
            zone_bc = zone["ZoneBC"]

            bc_names = set()
            for name in zone_bc:
                node = zone_bc[name]
                if isinstance(node, h5py.Group) and node.attrs.get("label", b"") == b"BC_t":
                    bc_names.add(name)

            assert len(bc_names) >= 6

    def test_cube3d_coordinates_exist(self, cube3d_path):
        with h5py.File(str(cube3d_path), "r") as f:
            zone = _get_cgns_zone(f)

            has_coords = False
            for name in zone:
                node = zone[name]
                if isinstance(node, h5py.Group) and node.attrs.get("label", b"") == b"GridCoordinates_t":
                    has_coords = True
                    coord_names = [k for k in node.keys() if k != " data"]
                    assert len(coord_names) >= 3
            assert has_coords
