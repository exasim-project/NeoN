# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os
import tempfile

import neon


def test_io_bindings_exist():
    assert hasattr(neon, 'write_vtm')
    assert hasattr(neon, 'write_cgns')
    assert hasattr(neon, 'write_vtk_hdf')
    assert hasattr(neon, 'read_cgns')
    assert hasattr(neon, 'read_vtk_hdf')


def test_write_vtm_creates_file():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 4, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "grid.vtm")
        neon.write_vtm(mesh, path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_write_cgns_creates_file():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 3, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "grid.cgns")
        neon.write_cgns(mesh, path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_write_vtk_hdf_creates_file():
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 2, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "grid.vtkhdf")
        neon.write_vtk_hdf(mesh, path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_cgns_roundtrip_preserves_boundaries():
    """Write CGNS, read back, verify 6 boundary patches survive."""
    exec = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec, 3, 2)
    assert mesh.n_boundaries() == 6

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roundtrip.cgns")
        neon.write_cgns(mesh, path)
        mesh2 = neon.read_cgns(path, exec)
        assert mesh2.n_boundaries() == 6
        assert mesh2.n_cells() == 6
