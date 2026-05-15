# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""End-to-end round-trip tests using the C++ cgnsRoundTrip CLI tool.

These tests read a reference CGNS mesh with pyvista, run the NeoN
reader+writer via the CLI tool, then compare the output with the
reference using pyvista.
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_roundtrip


class TestSingleTetRoundTrip:
    """Single tet round-trip through NeoN reader+writer."""

    def test_point_count_preserved(self, single_tet_path, roundtrip_tool, tmp_path):
        out = tmp_path / "singleTet_out.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        ref = extract_grid(single_tet_path)
        result = extract_grid(out)
        assert result.n_points == ref.n_points

    def test_tet_count_preserved(self, single_tet_path, roundtrip_tool, tmp_path):
        """Volume element count must match (ignoring boundary face elements)."""
        out = tmp_path / "singleTet_out.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        ref_tets = extract_grid(single_tet_path).extract_cells_by_type(10)
        out_tets = extract_grid(out).extract_cells_by_type(10)
        assert out_tets.n_cells == ref_tets.n_cells

    def test_points_preserved(self, single_tet_path, roundtrip_tool, tmp_path):
        out = tmp_path / "singleTet_out.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        ref = extract_grid(single_tet_path)
        result = extract_grid(out)
        np.testing.assert_allclose(
            np.sort(result.points, axis=0),
            np.sort(ref.points, axis=0),
            atol=1e-12,
        )

    def test_volume_is_one_sixth(self, single_tet_path, roundtrip_tool, tmp_path):
        out = tmp_path / "singleTet_out.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        result = extract_grid(out)
        tets = result.extract_cells_by_type(10)  # VTK_TETRA
        volumes = tets.compute_cell_sizes()["Volume"]
        assert len(volumes) == 1
        np.testing.assert_allclose(np.abs(volumes[0]), 1.0 / 6.0, rtol=1e-10)


class TestCube3DRoundTrip:
    """Cube 3D round-trip through NeoN reader+writer."""

    def test_point_count_preserved(self, cube3d_path, roundtrip_tool, tmp_path):
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref = extract_grid(cube3d_path)
        result = extract_grid(out)
        assert result.n_points == ref.n_points

    def test_tet_count_preserved(self, cube3d_path, roundtrip_tool, tmp_path):
        """Volume element (tet) count must match."""
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_tets = extract_grid(cube3d_path).extract_cells_by_type(10)
        out_tets = extract_grid(out).extract_cells_by_type(10)
        assert out_tets.n_cells == ref_tets.n_cells

    def test_points_preserved(self, cube3d_path, roundtrip_tool, tmp_path):
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref = extract_grid(cube3d_path)
        result = extract_grid(out)
        np.testing.assert_allclose(
            np.sort(result.points, axis=0),
            np.sort(ref.points, axis=0),
            atol=1e-12,
        )

    def test_total_volume_is_one(self, cube3d_path, roundtrip_tool, tmp_path):
        """Sum of absolute cell volumes must equal 1.0 (node orientation may differ)."""
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        result = extract_grid(out)
        tets = result.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        np.testing.assert_allclose(np.sum(np.abs(volumes)), 1.0, rtol=1e-10)

    def test_cell_volumes_match(self, cube3d_path, roundtrip_tool, tmp_path):
        """Sorted absolute cell volumes must match reference."""
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_tets = extract_grid(cube3d_path).extract_cells_by_type(10)
        out_tets = extract_grid(out).extract_cells_by_type(10)
        ref_vols = np.sort(np.abs(ref_tets.compute_cell_sizes()["Volume"]))
        out_vols = np.sort(np.abs(out_tets.compute_cell_sizes()["Volume"]))
        np.testing.assert_allclose(out_vols, ref_vols, rtol=1e-10)

    def test_surface_area_is_six(self, cube3d_path, roundtrip_tool, tmp_path):
        out = tmp_path / "cube3D_out.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        result = extract_grid(out)
        tets = result.extract_cells_by_type(10)
        surface = tets.extract_surface()
        areas = surface.compute_cell_sizes()["Area"]
        np.testing.assert_allclose(np.sum(areas), 6.0, rtol=1e-6)
