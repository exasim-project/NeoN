# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for mixed-element (hex+tet) mesh support.

Verifies that meshes containing multiple 3D element types (e.g. TETRA_4
and HEXA_8) can be read, written, and round-tripped correctly.
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_roundtrip


class TestMixedCellMesh:
    """Verify mixed-element mesh properties."""

    def test_has_multiple_element_types(self, mixed_grid):
        cell_types = set(mixed_grid.celltypes)
        # Should have at least 2 different 3D element types
        # VTK types: 10=TET, 12=HEX, 13=WEDGE, 14=PYRA
        volume_types = cell_types & {10, 12, 13, 14}
        assert len(volume_types) >= 2

    def test_has_cells(self, mixed_grid):
        assert mixed_grid.n_cells > 0

    def test_total_volume_is_one(self, mixed_grid):
        """Total volume should be 1.0 for unit cube."""
        # Extract all volume cell types
        volume_type_ids = [10, 12, 13, 14]
        total_vol = 0.0
        for vtype in volume_type_ids:
            cells = mixed_grid.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                total_vol += np.sum(np.abs(vols))
        np.testing.assert_allclose(total_vol, 1.0, rtol=1e-4)

    def test_all_volumes_positive(self, mixed_grid):
        volume_type_ids = [10, 12, 13, 14]
        for vtype in volume_type_ids:
            cells = mixed_grid.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                assert np.all(np.abs(vols) > 0), f"Zero volume cell of type {vtype}"

    def test_points_within_unit_cube(self, mixed_grid):
        pts = mixed_grid.points
        np.testing.assert_array_less(-1e-6, pts)
        np.testing.assert_array_less(pts, 1.0 + 1e-6)


class TestMixedCellRoundTrip:
    """Verify mixed-cell mesh survives NeoN round-trip."""

    def test_roundtrip_preserves_cell_count(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_out.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        ref = extract_grid(mixed_path)
        result = extract_grid(out)
        # Extract volume cells only
        ref_vol_cells = sum(
            ref.extract_cells_by_type(t).n_cells for t in [10, 12, 13, 14]
        )
        out_vol_cells = sum(
            result.extract_cells_by_type(t).n_cells for t in [10, 12, 13, 14]
        )
        assert out_vol_cells == ref_vol_cells

    def test_roundtrip_preserves_volume(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_out.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        result = extract_grid(out)
        total_vol = 0.0
        for vtype in [10, 12, 13, 14]:
            cells = result.extract_cells_by_type(vtype)
            if cells.n_cells > 0:
                vols = cells.compute_cell_sizes()["Volume"]
                total_vol += np.sum(np.abs(vols))
        np.testing.assert_allclose(total_vol, 1.0, rtol=1e-4)

    def test_roundtrip_preserves_point_count(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_out.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        ref = extract_grid(mixed_path)
        result = extract_grid(out)
        assert result.n_points == ref.n_points
