# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Fixtures for CGNS mesh IO tests."""

import pathlib
import subprocess

import pyvista as pv
import pytest

MESH_DIR = pathlib.Path(__file__).parent.parent / "meshFiles"
BUILD_DIR = (
    pathlib.Path(__file__).resolve().parents[5] / "build" / "develop" / "bin" / "tests"
)


def extract_grid(cgns_path):
    """Read CGNS file and extract the first UnstructuredGrid from the MultiBlock."""
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


@pytest.fixture
def mesh_dir():
    return MESH_DIR


@pytest.fixture
def single_tet_path(mesh_dir):
    p = mesh_dir / "singleTet.cgns"
    if not p.exists():
        pytest.skip(f"singleTet.cgns not found at {p}")
    return p


@pytest.fixture
def cube3d_path(mesh_dir):
    p = mesh_dir / "cube3D.cgns"
    if not p.exists():
        pytest.skip(f"cube3D.cgns not found at {p}")
    return p


@pytest.fixture
def cavity2d_path(mesh_dir):
    p = mesh_dir / "cavity2D.cgns"
    if not p.exists():
        pytest.skip(f"cavity2D.cgns not found at {p}")
    return p


@pytest.fixture
def single_tet_grid(single_tet_path):
    return extract_grid(single_tet_path)


@pytest.fixture
def cube3d_grid(cube3d_path):
    return extract_grid(cube3d_path)


@pytest.fixture
def cavity2d_grid(cavity2d_path):
    return extract_grid(cavity2d_path)


@pytest.fixture
def roundtrip_tool():
    """Path to the C++ cgnsRoundTrip CLI binary."""
    tool = BUILD_DIR / "cgnsRoundTrip"
    if not tool.exists():
        pytest.skip(f"cgnsRoundTrip not built at {tool}")
    return tool


@pytest.fixture
def vtu_tool():
    """Path to the C++ cgnsToVtu CLI binary."""
    tool = BUILD_DIR / "cgnsToVtu"
    if not tool.exists():
        pytest.skip(f"cgnsToVtu not built at {tool}")
    return tool


@pytest.fixture
def mixed_path(mesh_dir):
    p = mesh_dir / "mixedCells.cgns"
    if not p.exists():
        pytest.skip(f"mixedCells.cgns not found at {p}")
    return p


@pytest.fixture
def mixed_grid(mixed_path):
    return extract_grid(mixed_path)


def run_roundtrip(tool, input_path, output_path):
    """Run the C++ cgnsRoundTrip tool and assert it succeeds."""
    result = subprocess.run(
        [str(tool), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    fatal_lines = [
        line for line in result.stderr.splitlines()
        if line.strip() and "Mismatch in number of children" not in line
    ]
    assert result.returncode == 0, (
        f"cgnsRoundTrip failed (rc={result.returncode}):\n" + "\n".join(fatal_lines)
    )
    return result.stdout


def run_cgns_to_vtm(tool, input_path, output_path):
    """Run the C++ cgnsToVtu tool (writes VTM multiblock) and assert it succeeds."""
    result = subprocess.run(
        [str(tool), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    fatal_lines = [
        line for line in result.stderr.splitlines()
        if line.strip() and "Mismatch in number of children" not in line
    ]
    assert result.returncode == 0, (
        f"cgnsToVtu failed (rc={result.returncode}):\n" + "\n".join(fatal_lines)
    )
    return result.stdout


def extract_volume_block(vtm_path):
    """Read a VTM file and extract block 0 (the internalMesh / volume grid)."""
    mb = pv.read(str(vtm_path))
    if isinstance(mb, pv.MultiBlock):
        return mb[0]
    return mb
