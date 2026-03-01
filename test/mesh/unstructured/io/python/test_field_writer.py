# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for write_vtm / write_vtk_hdf with VolumeField data (Python bindings)."""

import numpy as np
import pyvista as pv
import pytest

import neon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mesh(exec):
    """2×2 uniform 2D hex mesh (4 cells, 6 boundary patches)."""
    return neon.create_uniform_2d_mesh(exec, nx=2, ny=2)


def make_scalar_field(exec, mesh):
    """Scalar VolumeField 'pressure' with internal value 1.0."""
    field = neon.ScalarVolumeField(exec, "pressure", mesh)
    neon.fill(field.internal_vector(), 1.0)
    field.correct_boundary_conditions()
    return field


def make_vec3_field(exec, mesh):
    """Vec3 VolumeField 'velocity' with internal value (2, 3, 4)."""
    field = neon.VectorVolumeField(exec, "velocity", mesh)
    neon.fill(field.internal_vector(), neon.Vec3(2.0, 3.0, 4.0))
    field.correct_boundary_conditions()
    return field


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exec():
    return neon.SerialExecutor()


@pytest.fixture
def mesh(exec):
    return make_mesh(exec)


@pytest.fixture
def scalar_field(exec, mesh):
    return make_scalar_field(exec, mesh)


@pytest.fixture
def vec3_field(exec, mesh):
    return make_vec3_field(exec, mesh)


# ---------------------------------------------------------------------------
# Scalar field — VTM
# ---------------------------------------------------------------------------

class TestScalarFieldVtm:
    def test_write_vtm_scalar_creates_file(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtm"
        neon.write_vtm(mesh, scalar_field, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_vtm_scalar_volume_has_pressure_array(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtm"
        neon.write_vtm(mesh, scalar_field, str(out))
        mb = pv.read(str(out))
        volume = mb[0]  # block 0 = internalMesh
        assert "pressure" in volume.cell_data

    def test_write_vtm_scalar_values_are_one(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtm"
        neon.write_vtm(mesh, scalar_field, str(out))
        mb = pv.read(str(out))
        values = mb[0].cell_data["pressure"]
        np.testing.assert_allclose(values, 1.0)

    def test_write_vtm_scalar_has_four_cells(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtm"
        neon.write_vtm(mesh, scalar_field, str(out))
        mb = pv.read(str(out))
        assert mb[0].cell_data["pressure"].shape[0] == 4

    def test_write_vtm_scalar_boundary_patches_have_array(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtm"
        neon.write_vtm(mesh, scalar_field, str(out))
        mb = pv.read(str(out))
        boundary = mb[1]  # block 1 = boundary multiblock
        for i in range(boundary.n_blocks):
            patch = boundary[i]
            assert patch is not None
            assert "pressure" in patch.cell_data


# ---------------------------------------------------------------------------
# Scalar field — VTK HDF
# ---------------------------------------------------------------------------

class TestScalarFieldVtkHdf:
    def test_write_vtk_hdf_scalar_creates_file(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "scalar.vtkhdf"
        neon.write_vtk_hdf(mesh, scalar_field, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Vec3 field — VTM
# ---------------------------------------------------------------------------

class TestVec3FieldVtm:
    def test_write_vtm_vec3_creates_file(self, mesh, vec3_field, tmp_path):
        out = tmp_path / "vector.vtm"
        neon.write_vtm(mesh, vec3_field, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_vtm_vec3_volume_has_velocity_array(self, mesh, vec3_field, tmp_path):
        out = tmp_path / "vector.vtm"
        neon.write_vtm(mesh, vec3_field, str(out))
        mb = pv.read(str(out))
        volume = mb[0]
        assert "velocity" in volume.cell_data

    def test_write_vtm_vec3_has_three_components(self, mesh, vec3_field, tmp_path):
        out = tmp_path / "vector.vtm"
        neon.write_vtm(mesh, vec3_field, str(out))
        mb = pv.read(str(out))
        arr = mb[0].cell_data["velocity"]
        assert arr.shape == (4, 3)


# ---------------------------------------------------------------------------
# Vec3 field — VTK HDF
# ---------------------------------------------------------------------------

class TestVec3FieldVtkHdf:
    def test_write_vtk_hdf_vec3_creates_file(self, mesh, vec3_field, tmp_path):
        out = tmp_path / "vector.vtkhdf"
        neon.write_vtk_hdf(mesh, vec3_field, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Multi-field — VTM
# ---------------------------------------------------------------------------

class TestMultiFieldVtm:
    def test_multi_field_vtm_creates_file(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "multi.vtm"
        fs = neon.FieldSet()
        fs.add_field(scalar_field).add_field(vec3_field)
        neon.write_vtm(mesh, fs, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multi_field_vtm_has_pressure(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "multi.vtm"
        fs = neon.FieldSet()
        fs.add_field(scalar_field).add_field(vec3_field)
        neon.write_vtm(mesh, fs, str(out))
        mb = pv.read(str(out))
        volume = mb[0]
        assert "pressure" in volume.cell_data

    def test_multi_field_vtm_has_velocity(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "multi.vtm"
        fs = neon.FieldSet()
        fs.add_field(scalar_field).add_field(vec3_field)
        neon.write_vtm(mesh, fs, str(out))
        mb = pv.read(str(out))
        volume = mb[0]
        assert "velocity" in volume.cell_data


# ---------------------------------------------------------------------------
# Multi-field — VTK HDF
# ---------------------------------------------------------------------------

class TestMultiFieldVtkHdf:
    def test_multi_field_vtkhdf_creates_file(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "multi.vtkhdf"
        fs = neon.FieldSet()
        fs.add_field(scalar_field).add_field(vec3_field)
        neon.write_vtk_hdf(mesh, fs, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# MeshWriter context manager
# ---------------------------------------------------------------------------

class TestMeshWriter:
    def test_context_manager_vtm(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "writer.vtm"
        with neon.MeshWriter(mesh, str(out)) as w:
            w.add_field(scalar_field)
            w.add_field(vec3_field)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_context_manager_vtkhdf(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "writer.vtkhdf"
        with neon.MeshWriter(mesh, str(out), fmt="vtkhdf") as w:
            w.add_field(scalar_field)
            w.add_field(vec3_field)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_context_manager_chaining(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "chain.vtm"
        with neon.MeshWriter(mesh, str(out)) as w:
            result = w.add_field(scalar_field)
            assert result is w
        assert out.exists()

    def test_context_manager_has_both_arrays(self, mesh, scalar_field, vec3_field, tmp_path):
        out = tmp_path / "both.vtm"
        with neon.MeshWriter(mesh, str(out)) as w:
            w.add_field(scalar_field)
            w.add_field(vec3_field)
        mb = pv.read(str(out))
        volume = mb[0]
        assert "pressure" in volume.cell_data
        assert "velocity" in volume.cell_data

    def test_context_manager_invalid_fmt(self, mesh, scalar_field, tmp_path):
        out = tmp_path / "bad.xyz"
        with pytest.raises(ValueError):
            with neon.MeshWriter(mesh, str(out), fmt="bad") as w:
                w.add_field(scalar_field)
