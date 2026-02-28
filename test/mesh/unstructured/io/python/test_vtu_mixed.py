# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""VTU writer tests for mixed-element meshes.

Verifies that the VTU export correctly handles meshes with multiple
3D element types (TETRA, HEXA, PYRAMID, WEDGE).
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_cgns_to_vtu


VOLUME_TYPES = [10, 12, 13, 14]  # TET, HEX, WEDGE, PYRA


class TestVtuMixedCells:
    """VTU output for mixed-element mesh."""

    def test_vtu_file_readable(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        grid = pv.read(str(out))
        assert grid is not None

    def test_vtu_has_multiple_element_types(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        grid = pv.read(str(out))
        vol_types = set(grid.celltypes) & set(VOLUME_TYPES)
        assert len(vol_types) >= 2

    def test_vtu_cell_count_matches_cgns(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        ref = extract_grid(mixed_path)
        result = pv.read(str(out))
        ref_vol = sum(ref.extract_cells_by_type(t).n_cells for t in VOLUME_TYPES)
        out_vol = sum(result.extract_cells_by_type(t).n_cells for t in VOLUME_TYPES)
        assert out_vol == ref_vol

    def test_vtu_point_count_matches_cgns(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        ref = extract_grid(mixed_path)
        result = pv.read(str(out))
        assert result.n_points == ref.n_points

    def test_vtu_total_volume_is_one(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        grid = pv.read(str(out))
        total_vol = 0.0
        for vtype in VOLUME_TYPES:
            cells = grid.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                total_vol += np.sum(np.abs(vols))
        np.testing.assert_allclose(total_vol, 1.0, rtol=1e-4)

    def test_vtu_points_within_unit_cube(self, mixed_path, vtu_tool, tmp_path):
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        grid = pv.read(str(out))
        np.testing.assert_array_less(-1e-6, grid.points)
        np.testing.assert_array_less(grid.points, 1.0 + 1e-6)

    def test_vtu_per_type_volumes_match_cgns(self, mixed_path, vtu_tool, tmp_path):
        """Per-element-type volume totals match between CGNS and VTU."""
        out = tmp_path / "mixed.vtu"
        run_cgns_to_vtu(vtu_tool, mixed_path, out)
        ref = extract_grid(mixed_path)
        result = pv.read(str(out))
        for vtype in VOLUME_TYPES:
            ref_cells = ref.extract_cells_by_type(vtype)
            out_cells = result.extract_cells_by_type(vtype)
            if ref_cells.n_cells == 0:
                continue
            ref_vol = np.sum(np.abs(ref_cells.compute_cell_sizes()["Volume"]))
            out_vol = np.sum(np.abs(out_cells.compute_cell_sizes()["Volume"]))
            np.testing.assert_allclose(
                out_vol, ref_vol, rtol=1e-4,
                err_msg=f"Volume mismatch for VTK type {vtype}",
            )
