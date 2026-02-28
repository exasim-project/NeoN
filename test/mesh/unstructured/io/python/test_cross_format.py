# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Cross-format consistency and double round-trip stability tests.

Verifies that:
1. CGNS round-trip output and VTU output produce equivalent geometry
2. Double CGNS round-trip (write -> read -> write -> read) is stable
"""

import numpy as np
import pyvista as pv
import pytest

from conftest import extract_grid, run_roundtrip, run_cgns_to_vtu


VOLUME_TYPES = [10, 12, 13, 14]  # TET, HEX, WEDGE, PYRA


def _total_volume(grid):
    """Sum of absolute volumes for all volume cell types."""
    total = 0.0
    for vtype in VOLUME_TYPES:
        cells = grid.extract_cells_by_type(vtype)
        if cells.n_cells > 0:
            vols = cells.compute_cell_sizes()["Volume"]
            total += np.sum(np.abs(vols))
    return total


def _volume_cell_count(grid):
    """Count of all volume cells."""
    return sum(grid.extract_cells_by_type(t).n_cells for t in VOLUME_TYPES)


class TestCrossFormatSingleTet:
    """CGNS round-trip vs VTU output consistency for single tet."""

    def test_point_count_consistent(
        self, single_tet_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "tet_rt.cgns"
        vtu_out = tmp_path / "tet.vtu"
        run_roundtrip(roundtrip_tool, single_tet_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, single_tet_path, vtu_out)
        cgns_grid = extract_grid(cgns_out)
        vtu_grid = pv.read(str(vtu_out))
        assert cgns_grid.n_points == vtu_grid.n_points

    def test_volume_consistent(
        self, single_tet_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "tet_rt.cgns"
        vtu_out = tmp_path / "tet.vtu"
        run_roundtrip(roundtrip_tool, single_tet_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, single_tet_path, vtu_out)
        cgns_vol = _total_volume(extract_grid(cgns_out))
        vtu_vol = _total_volume(pv.read(str(vtu_out)))
        np.testing.assert_allclose(cgns_vol, vtu_vol, rtol=1e-10)

    def test_points_consistent(
        self, single_tet_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "tet_rt.cgns"
        vtu_out = tmp_path / "tet.vtu"
        run_roundtrip(roundtrip_tool, single_tet_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, single_tet_path, vtu_out)
        cgns_pts = np.sort(extract_grid(cgns_out).points, axis=0)
        vtu_pts = np.sort(pv.read(str(vtu_out)).points, axis=0)
        np.testing.assert_allclose(cgns_pts, vtu_pts, atol=1e-12)


class TestCrossFormatCube3D:
    """CGNS round-trip vs VTU output consistency for cube 3D."""

    def test_point_count_consistent(
        self, cube3d_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "cube_rt.cgns"
        vtu_out = tmp_path / "cube.vtu"
        run_roundtrip(roundtrip_tool, cube3d_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, cube3d_path, vtu_out)
        cgns_grid = extract_grid(cgns_out)
        vtu_grid = pv.read(str(vtu_out))
        assert cgns_grid.n_points == vtu_grid.n_points

    def test_volume_consistent(
        self, cube3d_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "cube_rt.cgns"
        vtu_out = tmp_path / "cube.vtu"
        run_roundtrip(roundtrip_tool, cube3d_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, cube3d_path, vtu_out)
        cgns_vol = _total_volume(extract_grid(cgns_out))
        vtu_vol = _total_volume(pv.read(str(vtu_out)))
        np.testing.assert_allclose(cgns_vol, vtu_vol, rtol=1e-6)

    def test_sorted_volumes_consistent(
        self, cube3d_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "cube_rt.cgns"
        vtu_out = tmp_path / "cube.vtu"
        run_roundtrip(roundtrip_tool, cube3d_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, cube3d_path, vtu_out)
        cgns_tets = extract_grid(cgns_out).extract_cells_by_type(10)
        vtu_tets = pv.read(str(vtu_out)).extract_cells_by_type(10)
        cgns_vols = np.sort(np.abs(cgns_tets.compute_cell_sizes()["Volume"]))
        vtu_vols = np.sort(np.abs(vtu_tets.compute_cell_sizes()["Volume"]))
        np.testing.assert_allclose(cgns_vols, vtu_vols, rtol=1e-5)


class TestCrossFormatMixed:
    """CGNS round-trip vs VTU output consistency for mixed cells."""

    def test_cell_count_consistent(
        self, mixed_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "mixed_rt.cgns"
        vtu_out = tmp_path / "mixed.vtu"
        run_roundtrip(roundtrip_tool, mixed_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, mixed_path, vtu_out)
        cgns_count = _volume_cell_count(extract_grid(cgns_out))
        vtu_count = _volume_cell_count(pv.read(str(vtu_out)))
        assert cgns_count == vtu_count

    def test_volume_consistent(
        self, mixed_path, roundtrip_tool, vtu_tool, tmp_path
    ):
        cgns_out = tmp_path / "mixed_rt.cgns"
        vtu_out = tmp_path / "mixed.vtu"
        run_roundtrip(roundtrip_tool, mixed_path, cgns_out)
        run_cgns_to_vtu(vtu_tool, mixed_path, vtu_out)
        cgns_vol = _total_volume(extract_grid(cgns_out))
        vtu_vol = _total_volume(pv.read(str(vtu_out)))
        np.testing.assert_allclose(cgns_vol, vtu_vol, rtol=1e-4)


class TestDoubleRoundTripSingleTet:
    """Double CGNS round-trip stability for single tet."""

    def test_point_count_stable(
        self, single_tet_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "tet_rt1.cgns"
        rt2 = tmp_path / "tet_rt2.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        g1 = extract_grid(rt1)
        g2 = extract_grid(rt2)
        assert g1.n_points == g2.n_points

    def test_volume_stable(
        self, single_tet_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "tet_rt1.cgns"
        rt2 = tmp_path / "tet_rt2.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        vol1 = _total_volume(extract_grid(rt1))
        vol2 = _total_volume(extract_grid(rt2))
        np.testing.assert_allclose(vol2, vol1, rtol=1e-12)

    def test_points_stable(
        self, single_tet_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "tet_rt1.cgns"
        rt2 = tmp_path / "tet_rt2.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        pts1 = np.sort(extract_grid(rt1).points, axis=0)
        pts2 = np.sort(extract_grid(rt2).points, axis=0)
        np.testing.assert_allclose(pts2, pts1, atol=1e-12)


class TestDoubleRoundTripCube3D:
    """Double CGNS round-trip stability for cube 3D."""

    def test_point_count_stable(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "cube_rt1.cgns"
        rt2 = tmp_path / "cube_rt2.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        g1 = extract_grid(rt1)
        g2 = extract_grid(rt2)
        assert g1.n_points == g2.n_points

    def test_cell_count_stable(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "cube_rt1.cgns"
        rt2 = tmp_path / "cube_rt2.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        count1 = _volume_cell_count(extract_grid(rt1))
        count2 = _volume_cell_count(extract_grid(rt2))
        assert count1 == count2

    def test_volume_stable(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "cube_rt1.cgns"
        rt2 = tmp_path / "cube_rt2.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        vol1 = _total_volume(extract_grid(rt1))
        vol2 = _total_volume(extract_grid(rt2))
        np.testing.assert_allclose(vol2, vol1, rtol=1e-10)

    def test_sorted_volumes_stable(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "cube_rt1.cgns"
        rt2 = tmp_path / "cube_rt2.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        vols1 = np.sort(np.abs(
            extract_grid(rt1).extract_cells_by_type(10).compute_cell_sizes()["Volume"]
        ))
        vols2 = np.sort(np.abs(
            extract_grid(rt2).extract_cells_by_type(10).compute_cell_sizes()["Volume"]
        ))
        np.testing.assert_allclose(vols2, vols1, rtol=1e-10)


class TestDoubleRoundTripMixed:
    """Double CGNS round-trip stability for mixed-cell mesh."""

    def test_cell_count_stable(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "mixed_rt1.cgns"
        rt2 = tmp_path / "mixed_rt2.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        count1 = _volume_cell_count(extract_grid(rt1))
        count2 = _volume_cell_count(extract_grid(rt2))
        assert count1 == count2

    def test_volume_stable(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "mixed_rt1.cgns"
        rt2 = tmp_path / "mixed_rt2.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        vol1 = _total_volume(extract_grid(rt1))
        vol2 = _total_volume(extract_grid(rt2))
        np.testing.assert_allclose(vol2, vol1, rtol=1e-4)

    def test_point_count_stable(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        rt1 = tmp_path / "mixed_rt1.cgns"
        rt2 = tmp_path / "mixed_rt2.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, rt1)
        run_roundtrip(roundtrip_tool, rt1, rt2)
        g1 = extract_grid(rt1)
        g2 = extract_grid(rt2)
        assert g1.n_points == g2.n_points
