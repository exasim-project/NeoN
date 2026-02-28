# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Compare NeoN geometric quantities against pyvista computations.

Tests that cell volumes, cell centres, face areas, face normals, and
surface areas computed by NeoN (through round-trip CGNS) agree with
pyvista's independent computation from VTK.
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_roundtrip


class TestSingleTetGeometry:
    """Analytic geometry for reference tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)."""

    def test_volume(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        vol = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(vol[0], 1.0 / 6.0, rtol=1e-10)

    def test_centroid(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        centres = tets.cell_centers().points
        expected = np.array([[0.25, 0.25, 0.25]])
        np.testing.assert_allclose(centres, expected, atol=1e-10)

    def test_surface_area(self, single_tet_grid):
        tets = single_tet_grid.extract_cells_by_type(10)
        surface = tets.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        # 3 right triangles (area 0.5 each) + 1 equilateral-ish triangle
        expected_total = 3 * 0.5 + np.sqrt(3) / 2.0
        np.testing.assert_allclose(np.sum(areas), expected_total, rtol=1e-6)


class TestCube3DGeometry:
    """Geometric properties of the unit cube mesh."""

    def test_total_volume(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(np.sum(volumes), 1.0, rtol=1e-10)

    def test_all_volumes_positive(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        assert np.all(volumes > 0)

    def test_centroid_is_at_half(self, cube3d_grid):
        """Volume-weighted centroid of unit cube should be (0.5, 0.5, 0.5)."""
        tets = cube3d_grid.extract_cells_by_type(10)
        centres = tets.cell_centers().points
        volumes = tets.compute_cell_sizes()["Volume"]
        weighted_centre = np.average(centres, weights=volumes, axis=0)
        np.testing.assert_allclose(
            weighted_centre, [0.5, 0.5, 0.5], atol=1e-6
        )

    def test_surface_area(self, cube3d_grid):
        tets = cube3d_grid.extract_cells_by_type(10)
        surface = tets.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 6.0, rtol=1e-6)

    def test_face_normals_outward(self, cube3d_grid):
        """Surface face normals should point outward from the unit cube."""
        tets = cube3d_grid.extract_cells_by_type(10)
        surface = tets.extract_surface()
        normals = surface.cell_normals
        centres = surface.cell_centers().points

        cube_centre = np.array([0.5, 0.5, 0.5])
        outward = centres - cube_centre
        dots = np.sum(normals * outward, axis=1)
        assert np.all(dots >= -1e-10)


class TestCube3DRoundTripGeometry:
    """Geometric quantities preserved through NeoN round-trip."""

    def test_cell_centres_preserved(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_geom.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)

        ref = extract_grid(cube3d_path).extract_cells_by_type(10)
        result = extract_grid(out).extract_cells_by_type(10)

        ref_centres = np.sort(ref.cell_centers().points, axis=0)
        out_centres = np.sort(result.cell_centers().points, axis=0)
        np.testing.assert_allclose(out_centres, ref_centres, atol=1e-6)

    def test_weighted_centroid_preserved(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_geom.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        result = extract_grid(out).extract_cells_by_type(10)
        centres = result.cell_centers().points
        volumes = np.abs(result.compute_cell_sizes()["Volume"])
        weighted = np.average(centres, weights=volumes, axis=0)
        np.testing.assert_allclose(weighted, [0.5, 0.5, 0.5], atol=1e-4)

    def test_surface_area_preserved(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_geom.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        result = extract_grid(out).extract_cells_by_type(10)
        surface = result.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 6.0, rtol=1e-4)

    def test_per_cell_volumes_preserved(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_geom.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)

        ref = extract_grid(cube3d_path).extract_cells_by_type(10)
        result = extract_grid(out).extract_cells_by_type(10)

        ref_vols = np.sort(np.abs(ref.compute_cell_sizes()["Volume"]))
        out_vols = np.sort(np.abs(result.compute_cell_sizes()["Volume"]))
        np.testing.assert_allclose(out_vols, ref_vols, rtol=1e-5)


class TestCavity2DGeometry:
    """Geometric properties of the cavity 2D mesh."""

    def test_bounding_box(self, cavity2d_grid):
        bounds = cavity2d_grid.bounds
        # Cavity should have finite extent in x and y
        assert bounds[1] - bounds[0] > 0  # x extent
        assert bounds[3] - bounds[2] > 0  # y extent

    def test_all_points_finite(self, cavity2d_grid):
        assert np.all(np.isfinite(cavity2d_grid.points))

    def test_roundtrip_bounding_box_preserved(
        self, cavity2d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cavity2D_geom.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = extract_grid(out)
        np.testing.assert_allclose(result.bounds, ref.bounds, atol=1e-12)

    def test_roundtrip_points_preserved(
        self, cavity2d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cavity2D_geom.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = extract_grid(out)
        np.testing.assert_allclose(
            np.sort(result.points, axis=0),
            np.sort(ref.points, axis=0),
            atol=1e-12,
        )


class TestMixedCellGeometry:
    """Geometric properties of the mixed-cell mesh."""

    def test_total_volume_is_one(self, mixed_grid):
        volume_types = [10, 12, 13, 14]
        total_vol = 0.0
        for vtype in volume_types:
            cells = mixed_grid.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                total_vol += np.sum(np.abs(vols))
        np.testing.assert_allclose(total_vol, 1.0, rtol=1e-4)

    def test_weighted_centroid(self, mixed_grid):
        """Volume-weighted centroid should be near (0.5, 0.5, 0.5)."""
        volume_types = [10, 12, 13, 14]
        all_centres = []
        all_volumes = []
        for vtype in volume_types:
            cells = mixed_grid.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                centres = cells.cell_centers().points
                volumes = np.abs(cells.compute_cell_sizes()["Volume"])
                all_centres.append(centres)
                all_volumes.append(volumes)
        centres = np.vstack(all_centres)
        volumes = np.concatenate(all_volumes)
        weighted = np.average(centres, weights=volumes, axis=0)
        np.testing.assert_allclose(weighted, [0.5, 0.5, 0.5], atol=0.1)

    def test_roundtrip_volume_preserved(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_geom.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        result = extract_grid(out)
        volume_types = [10, 12, 13, 14]
        total_vol = 0.0
        for vtype in volume_types:
            cells = result.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                total_vol += np.sum(np.abs(vols))
        np.testing.assert_allclose(total_vol, 1.0, rtol=1e-4)

    def test_roundtrip_points_within_bounds(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_geom.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        result = extract_grid(out)
        np.testing.assert_array_less(-1e-6, result.points)
        np.testing.assert_array_less(result.points, 1.0 + 1e-6)
