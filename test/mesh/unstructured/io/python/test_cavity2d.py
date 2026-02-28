# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cavity 2D mesh round-trip and VTU tests.

Tests the 2D cavity mesh through CGNS round-trip and VTU export paths.
NeoN currently only supports 3D cell types (tet, hex, wedge, pyramid),
so round-trip tests for 2D triangle meshes are marked xfail.
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_roundtrip, run_cgns_to_vtu


class TestCavity2DReference:
    """Verify the reference cavity2D mesh is valid."""

    def test_has_triangles(self, cavity2d_grid):
        tris = cavity2d_grid.extract_cells_by_type(5)  # VTK_TRIANGLE
        assert tris.n_cells > 0

    def test_all_points_finite(self, cavity2d_grid):
        assert np.all(np.isfinite(cavity2d_grid.points))

    def test_bounding_box_has_extent(self, cavity2d_grid):
        bounds = cavity2d_grid.bounds
        assert bounds[1] - bounds[0] > 0  # x extent
        assert bounds[3] - bounds[2] > 0  # y extent


class TestCavity2DRoundTrip:
    """Cavity 2D CGNS round-trip through NeoN reader+writer."""

    def test_point_count_preserved(self, cavity2d_path, roundtrip_tool, tmp_path):
        out = tmp_path / "cavity2D_out.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = extract_grid(out)
        assert result.n_points == ref.n_points

    @pytest.mark.xfail(reason="NeoN only supports 3D cell types; 2D triangles not round-tripped")
    def test_cell_count_preserved(self, cavity2d_path, roundtrip_tool, tmp_path):
        out = tmp_path / "cavity2D_out.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = extract_grid(out)
        ref_vol = sum(ref.extract_cells_by_type(t).n_cells for t in [5, 7, 9, 10, 12])
        out_vol = sum(
            result.extract_cells_by_type(t).n_cells for t in [5, 7, 9, 10, 12]
        )
        assert out_vol == ref_vol


class TestCavity2DVtu:
    """Cavity 2D VTU export tests."""

    def test_vtu_point_count_matches(self, cavity2d_path, vtu_tool, tmp_path):
        out = tmp_path / "cavity2D.vtu"
        run_cgns_to_vtu(vtu_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = pv.read(str(out))
        assert result.n_points == ref.n_points

    @pytest.mark.xfail(reason="NeoN only supports 3D cell types; 2D triangles not exported to VTU")
    def test_vtu_cell_count_matches(self, cavity2d_path, vtu_tool, tmp_path):
        out = tmp_path / "cavity2D.vtu"
        run_cgns_to_vtu(vtu_tool, cavity2d_path, out)
        ref = extract_grid(cavity2d_path)
        result = pv.read(str(out))
        ref_cells = sum(ref.extract_cells_by_type(t).n_cells for t in [5, 7, 9, 10, 12])
        out_cells = sum(
            result.extract_cells_by_type(t).n_cells for t in [5, 7, 9, 10, 12]
        )
        assert out_cells == ref_cells
