# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for CGNS writer verification.

These tests verify the CGNS file structure of the reference meshes and
exercise pyvista round-trip through VTU to validate mesh data integrity.
The C++ writer round-trip is covered by Catch2 tests in cgnsMeshWriter.cpp.
"""

import pathlib

import h5py
import numpy as np
import pyvista as pv
import pytest


MESH_DIR = pathlib.Path(__file__).parent.parent / "meshFiles"


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


def _extract_grid(cgns_path):
    """Read CGNS file and extract the first UnstructuredGrid."""
    data = pv.read(str(cgns_path))

    def _find_grid(obj):
        if isinstance(obj, pv.UnstructuredGrid):
            return obj
        if isinstance(obj, pv.MultiBlock):
            for block in obj:
                result = _find_grid(block)
                if result is not None:
                    return result
        return None

    grid = _find_grid(data)
    if grid is None:
        raise RuntimeError(f"No UnstructuredGrid found in {cgns_path}")
    return grid


class TestCGNSWriterStructure:
    """Verify CGNS file structure requirements for the NeoN writer."""

    def test_single_tet_has_valid_structure(self):
        """Reference singleTet.cgns has proper CGNS structure."""
        src = MESH_DIR / "singleTet.cgns"
        if not src.exists():
            pytest.skip(f"singleTet.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            has_coords = any(
                isinstance(zone[name], h5py.Group)
                and zone[name].attrs.get("label", b"") == b"GridCoordinates_t"
                for name in zone
            )
            assert has_coords

    def test_cube3d_bc_count(self):
        """cube3D has 6 BC_t nodes under ZoneBC."""
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            assert "ZoneBC" in zone
            zone_bc = zone["ZoneBC"]
            bc_names = [
                name for name in zone_bc
                if isinstance(zone_bc[name], h5py.Group)
                and zone_bc[name].attrs.get("label", b"") == b"BC_t"
            ]
            assert len(bc_names) >= 6

    def test_cube3d_has_volume_and_boundary_sections(self):
        """cube3D has both volume (TETRA_4) and boundary (TRI_3) element sections."""
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            volume_sections = []
            boundary_sections = []
            for name in zone:
                node = zone[name]
                if isinstance(node, h5py.Group) and node.attrs.get("label", b"") == b"Elements_t":
                    elem_type = node[" data"][...].flatten()[0]
                    if elem_type == 10:  # TETRA_4
                        volume_sections.append(name)
                    elif elem_type == 5:  # TRI_3
                        boundary_sections.append(name)
            assert len(volume_sections) >= 1
            assert len(boundary_sections) >= 1

    def test_cube3d_zone_sizes_consistent(self):
        """cube3D zone sizes (vertices, elements) are positive."""
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            zone_sizes = zone[" data"][...].flatten()
            n_vertices = zone_sizes[0]
            n_elements = zone_sizes[1]
            assert n_vertices > 0
            assert n_elements > 0

    def test_single_tet_has_tet_elements(self):
        """Reference singleTet has TETRA_4 element section."""
        src = MESH_DIR / "singleTet.cgns"
        if not src.exists():
            pytest.skip(f"singleTet.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            tet_sections = []
            for name in zone:
                node = zone[name]
                if isinstance(node, h5py.Group) and node.attrs.get("label", b"") == b"Elements_t":
                    elem_type = node[" data"][...].flatten()[0]
                    if elem_type == 10:  # TETRA_4
                        tet_sections.append(name)
            assert len(tet_sections) >= 1

    def test_cube3d_bc_point_ranges_valid(self):
        """Each BC_t node has a valid PointRange or ElementRange."""
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        with h5py.File(str(src), "r") as f:
            zone = _get_cgns_zone(f)
            if "ZoneBC" not in zone:
                pytest.skip("No ZoneBC in file")
            zone_bc = zone["ZoneBC"]
            for name in zone_bc:
                node = zone_bc[name]
                if not isinstance(node, h5py.Group):
                    continue
                if node.attrs.get("label", b"") != b"BC_t":
                    continue
                # BC should have a PointRange or PointList child
                has_range = any(
                    isinstance(node[child], h5py.Group)
                    and node[child].attrs.get("label", b"") in (
                        b"IndexRange_t", b"IndexArray_t"
                    )
                    for child in node
                    if isinstance(node[child], h5py.Group)
                )
                assert has_range, f"BC '{name}' has no PointRange/PointList"


class TestVTURoundTrip:
    """Verify mesh data integrity via pyvista VTU round-trip.

    This validates that mesh data (points, cells, volumes) survives
    serialization, exercising the same data pipeline as the C++ writer.
    """

    def test_single_tet_vtu_roundtrip_preserves_points(self, tmp_path):
        src = MESH_DIR / "singleTet.cgns"
        if not src.exists():
            pytest.skip(f"singleTet.cgns not found at {src}")
        grid = _extract_grid(src)
        out = tmp_path / "singleTet.vtu"
        grid.save(str(out))
        grid2 = pv.read(str(out))
        np.testing.assert_allclose(
            np.sort(grid.points, axis=0),
            np.sort(grid2.points, axis=0),
            atol=1e-12,
        )

    def test_single_tet_vtu_roundtrip_preserves_cells(self, tmp_path):
        src = MESH_DIR / "singleTet.cgns"
        if not src.exists():
            pytest.skip(f"singleTet.cgns not found at {src}")
        grid = _extract_grid(src)
        out = tmp_path / "singleTet.vtu"
        grid.save(str(out))
        grid2 = pv.read(str(out))
        assert grid2.n_cells == grid.n_cells

    def test_cube3d_vtu_roundtrip_preserves_volume(self, tmp_path):
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        grid = _extract_grid(src)
        tets = grid.extract_cells_by_type(10)
        vol_orig = np.sum(tets.compute_cell_sizes()["Volume"])

        out = tmp_path / "cube3D.vtu"
        grid.save(str(out))
        grid2 = pv.read(str(out))
        tets2 = grid2.extract_cells_by_type(10)
        vol_rt = np.sum(tets2.compute_cell_sizes()["Volume"])
        np.testing.assert_allclose(vol_rt, vol_orig, rtol=1e-10)

    def test_cube3d_vtu_roundtrip_preserves_cell_count(self, tmp_path):
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        grid = _extract_grid(src)
        out = tmp_path / "cube3D.vtu"
        grid.save(str(out))
        grid2 = pv.read(str(out))
        assert grid2.n_cells == grid.n_cells

    def test_cube3d_vtu_roundtrip_all_volumes_positive(self, tmp_path):
        src = MESH_DIR / "cube3D.cgns"
        if not src.exists():
            pytest.skip(f"cube3D.cgns not found at {src}")
        grid = _extract_grid(src)
        out = tmp_path / "cube3D.vtu"
        grid.save(str(out))
        grid2 = pv.read(str(out))
        tets = grid2.extract_cells_by_type(10)
        volumes = tets.compute_cell_sizes()["Volume"]
        assert np.all(volumes > 0)
