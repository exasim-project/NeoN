# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the VTM (multi-block) writer round-trip verification.

The C++ cgnsToVtu tool reads a CGNS mesh with NeoN and writes it as VTM.
These tests verify the output is valid and preserves mesh properties.
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, extract_volume_block, run_cgns_to_vtm


class TestVtmSingleTet:
    """Verify VTM output for the single tet mesh."""

    def test_vtm_file_readable(self, single_tet_path, vtu_tool, tmp_path):
        out = tmp_path / "singleTet.vtm"
        run_cgns_to_vtm(vtu_tool, single_tet_path, out)
        data = pv.read(str(out))
        assert data is not None

    def test_vtm_has_one_tet(self, single_tet_path, vtu_tool, tmp_path):
        out = tmp_path / "singleTet.vtm"
        run_cgns_to_vtm(vtu_tool, single_tet_path, out)
        grid = extract_volume_block(out)
        tets = grid.extract_cells_by_type(10)
        assert tets.n_cells == 1

    def test_vtm_has_four_points(self, single_tet_path, vtu_tool, tmp_path):
        out = tmp_path / "singleTet.vtm"
        run_cgns_to_vtm(vtu_tool, single_tet_path, out)
        grid = extract_volume_block(out)
        assert grid.n_points == 4

    def test_vtm_volume_is_one_sixth(self, single_tet_path, vtu_tool, tmp_path):
        out = tmp_path / "singleTet.vtm"
        run_cgns_to_vtm(vtu_tool, single_tet_path, out)
        grid = extract_volume_block(out)
        tets = grid.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(np.abs(volumes[0]), 1.0 / 6.0, rtol=1e-10)

    def test_vtm_points_match_reference(self, single_tet_path, vtu_tool, tmp_path):
        out = tmp_path / "singleTet.vtm"
        run_cgns_to_vtm(vtu_tool, single_tet_path, out)
        ref = extract_grid(single_tet_path)
        result = extract_volume_block(out)
        np.testing.assert_allclose(
            np.sort(result.points, axis=0),
            np.sort(ref.points, axis=0),
            atol=1e-12,
        )


class TestVtmCube3D:
    """Verify VTM output for the cube 3D mesh."""

    def test_vtm_file_readable(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        data = pv.read(str(out))
        assert data is not None

    def test_vtm_tet_count_matches(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        ref_tets = extract_grid(cube3d_path).extract_cells_by_type(10)
        result = extract_volume_block(out)
        out_tets = result.extract_cells_by_type(10)
        assert out_tets.n_cells == ref_tets.n_cells

    def test_vtm_point_count_matches(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        ref = extract_grid(cube3d_path)
        result = extract_volume_block(out)
        assert result.n_points == ref.n_points

    def test_vtm_total_volume_is_one(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        result = extract_volume_block(out)
        tets = result.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(np.sum(np.abs(volumes)), 1.0, rtol=1e-10)

    def test_vtm_points_within_unit_cube(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        result = extract_volume_block(out)
        np.testing.assert_array_less(-1e-10, result.points)
        np.testing.assert_array_less(result.points, 1.0 + 1e-10)

    def test_vtm_surface_area_is_six(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        result = extract_volume_block(out)
        tets = result.extract_cells_by_type(10)
        surface = tets.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 6.0, rtol=1e-6)

    def test_vtm_cell_volumes_match_reference(self, cube3d_path, vtu_tool, tmp_path):
        out = tmp_path / "cube3D.vtm"
        run_cgns_to_vtm(vtu_tool, cube3d_path, out)
        ref_tets = extract_grid(cube3d_path).extract_cells_by_type(10)
        out_tets = extract_volume_block(out).extract_cells_by_type(10)
        ref_vols = np.sort(np.abs(ref_tets.compute_cell_sizes()["Volume"]))
        out_vols = np.sort(np.abs(out_tets.compute_cell_sizes()["Volume"]))
        np.testing.assert_allclose(out_vols, ref_vols, rtol=1e-5)
