# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Boundary patch name and face count preservation tests.

Verifies that named boundary patches survive the full round-trip:
gmsh -> CGNS -> NeoN reader -> NeoN writer -> CGNS -> h5py inspection.
"""

import h5py
import numpy as np
import pytest

from conftest import run_roundtrip


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


def _get_bc_names(cgns_path, surface_only=True):
    """Extract BC_t node names from a CGNS file via h5py.

    If surface_only=True, exclude volume (V_*), line (L_*), and point (P_*)
    BCs that gmsh generates but NeoN doesn't use.
    """
    with h5py.File(str(cgns_path), "r") as f:
        zone = _get_cgns_zone(f)
        if "ZoneBC" not in zone:
            return []
        zone_bc = zone["ZoneBC"]
        names = sorted(
            name for name in zone_bc
            if isinstance(zone_bc[name], h5py.Group)
            and zone_bc[name].attrs.get("label", b"") == b"BC_t"
        )
        if surface_only:
            names = [n for n in names if n.startswith("S_")]
        return names


def _get_bc_face_counts(cgns_path, surface_only=True):
    """Get face count per BC by inspecting PointRange in each BC_t node."""
    counts = {}
    with h5py.File(str(cgns_path), "r") as f:
        zone = _get_cgns_zone(f)
        if "ZoneBC" not in zone:
            return counts
        zone_bc = zone["ZoneBC"]
        for name in zone_bc:
            if surface_only and not name.startswith("S_"):
                continue
            node = zone_bc[name]
            if not isinstance(node, h5py.Group):
                continue
            if node.attrs.get("label", b"") != b"BC_t":
                continue
            # Look for PointRange child
            for child_name in node:
                child = node[child_name]
                if isinstance(child, h5py.Group) and child.attrs.get("label", b"") == b"IndexRange_t":
                    data = child[" data"][...].flatten()
                    counts[name] = int(data[1] - data[0] + 1)
                    break
    return counts


def _get_boundary_section_counts(cgns_path):
    """Get element count per boundary section from Elements_t nodes."""
    counts = {}
    with h5py.File(str(cgns_path), "r") as f:
        zone = _get_cgns_zone(f)
        for name in zone:
            node = zone[name]
            if not isinstance(node, h5py.Group):
                continue
            if node.attrs.get("label", b"") != b"Elements_t":
                continue
            elem_type = node[" data"][...].flatten()[0]
            # TRI_3=5 for 3D boundary, BAR_2=3 for 2D boundary
            if elem_type in (3, 5):
                # ElementRange gives start..end
                for child_name in node:
                    child = node[child_name]
                    if isinstance(child, h5py.Group) and child.attrs.get("label", b"") == b"IndexRange_t":
                        data = child[" data"][...].flatten()
                        counts[name] = int(data[1] - data[0] + 1)
                        break
    return counts


class TestCube3DBCNamePreservation:
    """Verify that cube3D boundary patch names survive round-trip."""

    def test_reference_has_six_bc_names(self, cube3d_path):
        names = _get_bc_names(cube3d_path)
        assert len(names) == 6

    def test_roundtrip_preserves_bc_count(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_bc.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_names = _get_bc_names(cube3d_path)
        out_names = _get_bc_names(out)
        assert len(out_names) == len(ref_names)

    def test_roundtrip_preserves_bc_names(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_bc.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_names = _get_bc_names(cube3d_path)
        out_names = _get_bc_names(out)
        assert out_names == ref_names

    def test_roundtrip_preserves_bc_face_counts(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_bc.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_counts = _get_bc_face_counts(cube3d_path)
        out_counts = _get_bc_face_counts(out)
        assert ref_counts == out_counts

    def test_roundtrip_preserves_boundary_section_counts(
        self, cube3d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cube3D_bc.cgns"
        run_roundtrip(roundtrip_tool, cube3d_path, out)
        ref_counts = _get_boundary_section_counts(cube3d_path)
        out_counts = _get_boundary_section_counts(out)
        # Total face count per section should match
        assert sum(ref_counts.values()) == sum(out_counts.values())


class TestSingleTetBCPreservation:
    """Verify boundary patches for the single tet mesh."""

    def test_reference_has_bc_names(self, single_tet_path):
        names = _get_bc_names(single_tet_path)
        assert len(names) == 4  # 4 surface patches (S_1..S_4)

    def test_roundtrip_preserves_bc_count(
        self, single_tet_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "singleTet_bc.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        ref_names = _get_bc_names(single_tet_path)
        out_names = _get_bc_names(out)
        assert len(out_names) == len(ref_names)

    def test_roundtrip_preserves_bc_names(
        self, single_tet_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "singleTet_bc.cgns"
        run_roundtrip(roundtrip_tool, single_tet_path, out)
        ref_names = _get_bc_names(single_tet_path)
        out_names = _get_bc_names(out)
        assert out_names == ref_names


class TestCavity2DBCPreservation:
    """Verify boundary patches for the cavity 2D mesh."""

    def test_reference_has_bc_names(self, cavity2d_path):
        names = _get_bc_names(cavity2d_path)
        assert len(names) >= 1

    @pytest.mark.xfail(reason="NeoN only supports 3D cell types; 2D BC round-trip not implemented")
    def test_roundtrip_preserves_bc_count(
        self, cavity2d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cavity2D_bc.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref_names = _get_bc_names(cavity2d_path)
        out_names = _get_bc_names(out)
        assert len(out_names) == len(ref_names)

    @pytest.mark.xfail(reason="NeoN only supports 3D cell types; 2D BC round-trip not implemented")
    def test_roundtrip_preserves_bc_names(
        self, cavity2d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cavity2D_bc.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref_names = _get_bc_names(cavity2d_path)
        out_names = _get_bc_names(out)
        assert out_names == ref_names

    @pytest.mark.xfail(reason="NeoN only supports 3D cell types; 2D BC round-trip not implemented")
    def test_roundtrip_preserves_bc_face_counts(
        self, cavity2d_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "cavity2D_bc.cgns"
        run_roundtrip(roundtrip_tool, cavity2d_path, out)
        ref_counts = _get_bc_face_counts(cavity2d_path)
        out_counts = _get_bc_face_counts(out)
        assert ref_counts == out_counts


class TestMixedCellBCPreservation:
    """Verify boundary patches for the mixed-cell mesh."""

    def test_reference_has_bc_names(self, mixed_path):
        names = _get_bc_names(mixed_path)
        assert len(names) >= 1

    def test_roundtrip_preserves_bc_count(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_bc.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        ref_names = _get_bc_names(mixed_path)
        out_names = _get_bc_names(out)
        assert len(out_names) == len(ref_names)

    def test_roundtrip_preserves_bc_names(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_bc.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        ref_names = _get_bc_names(mixed_path)
        out_names = _get_bc_names(out)
        assert out_names == ref_names

    def test_roundtrip_preserves_bc_face_counts(
        self, mixed_path, roundtrip_tool, tmp_path
    ):
        out = tmp_path / "mixed_bc.cgns"
        run_roundtrip(roundtrip_tool, mixed_path, out)
        ref_counts = _get_bc_face_counts(mixed_path)
        out_counts = _get_bc_face_counts(out)
        assert ref_counts == out_counts
