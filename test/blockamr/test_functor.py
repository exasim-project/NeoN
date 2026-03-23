# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TDD tests for NamedTuple functor pattern (build_kernel)."""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import Field, CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.div import Div, build_face_fluxes
from neon.blockamr.operators.laplacian import Laplacian
from neon.blockamr.operators.grad import Grad
from neon.blockamr.operators.source import Source
from neon.blockamr.schemes.div_schemes import Upwind, Linear, VanLeer, QUICK
from neon.blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from neon.blockamr.schemes.grad_schemes import CentralDiffGrad
from neon.blockamr.operators.div import BoxFluxData


def _make_mesh(n_cell=32, max_size=32):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh, box, dm, geom


def _make_field(n_cell=32, max_size=32, ngrow=1, name="phi"):
    """Create a periodic Field on [0,1]^3 (low-level, for scheme tests)."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom, name=name, box=box, dm=dm, max_size=max_size), box, dm, geom


def _init_sin3d_cell(phi, geom):
    """Set CellField to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)


def _init_sin3d(field):
    """Set Field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) (low-level)."""
    dx = field.dx
    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
        field.mf.copy_from(mfi, arr)
    field.fill_boundary()


def _make_fluxes(box, dm, geom, ngrow=1):
    """Build a FaceField for operator tests."""
    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    return build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0)


def _make_fluxes_level(box, dm, geom, ngrow=1):
    """Build a _FaceFieldLevel for scheme-level tests."""
    return _make_fluxes(box, dm, geom, ngrow=ngrow)[0]


# ---------------------------------------------------------------------------
# FaceField DistributionMapping test
# ---------------------------------------------------------------------------


class TestFaceFieldSharesDm:
    def test_cross_multifab_access_via_shared_mfiter(self):
        """FaceField must share caller's dm so MFIterator works across MultiFabs."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        ff = FaceField(mesh, ncomp=1, ngrow=1)
        for mfi in blockamr.MFIterator(phi.mf[0]):
            for d in range(3):
                arr = ff[0][d].mf.grown_array(mfi)
                assert arr.shape[0] > 0


# ---------------------------------------------------------------------------
# Operator build_kernel(mfi, t) tests
# ---------------------------------------------------------------------------


class TestDivBuildKernelMfi:
    def test_returns_callable(self):
        """Div.build_kernel(mfi, t) returns a callable."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        for mfi in blockamr.MFIterator(phi.mf[0]):
            kernel = div_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestLaplacianBuildKernelMfi:
    def test_returns_callable(self):
        mesh, *_ = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, mesh.geom(0))
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), phi)
        for mfi in blockamr.MFIterator(phi.mf[0]):
            kernel = lap_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestGradBuildKernelMfi:
    def test_returns_callable(self):
        mesh, *_ = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        grad_op = Grad(phi)
        for mfi in blockamr.MFIterator(phi.mf[0]):
            kernel = grad_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestSourceBuildKernelMfi:
    def test_returns_callable(self):
        mesh, *_ = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        src_op = Source(lambda x, y, z, t: np.ones_like(x), phi)
        for mfi in blockamr.MFIterator(phi.mf[0]):
            kernel = src_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestFusedStepMfi:
    def test_fused_step_with_mfi(self):
        """_fused_step with mfi-based kernels matches sequential sum."""
        from neon.blockamr.dsl.solve import _fused_step

        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), phi, coeff=-1.0)

        for mfi in blockamr.MFIterator(phi.mf[0]):
            phi_arr = jnp.asarray(phi.mf[0].grown_array(mfi))
            k_div = div_op.build_kernel(mfi, t=0.0)
            k_lap = lap_op.build_kernel(mfi, t=0.0)
            fused = _fused_step(phi_arr, [k_div, k_lap])
            sequential = k_div(phi_arr) + k_lap(phi_arr)
            assert jnp.allclose(fused, sequential, atol=1e-12)


# ---------------------------------------------------------------------------
# Scheme-level build_kernel tests (use low-level Field + _FaceFieldLevel)
# ---------------------------------------------------------------------------


def _make_box_flux_data(face_fluxes_lev, mfi, geom, stencil_width, ngrow=0):
    """Build a BoxFluxData from a _FaceFieldLevel and MFIterator."""
    return BoxFluxData(
        flux_x=face_fluxes_lev[0].mf.grown_array(mfi),
        flux_y=face_fluxes_lev[1].mf.grown_array(mfi),
        flux_z=face_fluxes_lev[2].mf.grown_array(mfi),
        dh=jnp.array(geom.cell_size()),
        stencil_width=stencil_width,
        ngrow=ngrow if ngrow > 0 else stencil_width,
    )


class TestUpwindSchemeBuildKernel:
    def test_returns_named_tuple(self):
        """Upwind.build_kernel() returns a NamedTuple."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=1)
        scheme = Upwind()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            assert isinstance(kernel, tuple)
            assert hasattr(kernel, '_fields')

    def test_callable(self):
        """Upwind kernel functor is callable with phi."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=1)
        scheme = Upwind()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            phi_arr = phi.mf[0].grown_array(mfi)
            result = kernel(phi_arr)
            assert result.ndim == 3

    def test_matches_compute(self):
        """Upwind build_kernel() output matches compute()."""
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=1)
        scheme = Upwind()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            phi_arr = phi.mf[0].grown_array(mfi)
            result = kernel(phi_arr)
            phi_3d = phi_arr[:, :, :, 0]
            fx = flux_data.flux_x[:, :, :, 0]
            fy = flux_data.flux_y[:, :, :, 0]
            fz = flux_data.flux_z[:, :, :, 0]
            expected = scheme.compute(phi_3d, [fx, fy, fz], flux_data.dh)
            assert jnp.allclose(result, expected, atol=1e-12)


class TestLinearSchemeBuildKernel:
    def test_matches_compute(self):
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=1)
        scheme = Linear()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            phi_arr = phi.mf[0].grown_array(mfi)
            result = kernel(phi_arr)
            phi_3d = phi_arr[:, :, :, 0]
            fx = flux_data.flux_x[:, :, :, 0]
            fy = flux_data.flux_y[:, :, :, 0]
            fz = flux_data.flux_z[:, :, :, 0]
            expected = scheme.compute(phi_3d, [fx, fy, fz], flux_data.dh)
            assert jnp.allclose(result, expected, atol=1e-12)


class TestVanLeerSchemeBuildKernel:
    def test_matches_compute(self):
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=2, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=2)
        scheme = VanLeer()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            phi_arr = phi.mf[0].grown_array(mfi)
            result = kernel(phi_arr)
            phi_3d = phi_arr[:, :, :, 0]
            fx = flux_data.flux_x[:, :, :, 0]
            fy = flux_data.flux_y[:, :, :, 0]
            fz = flux_data.flux_z[:, :, :, 0]
            expected = scheme.compute(phi_3d, [fx, fy, fz], flux_data.dh)
            assert jnp.allclose(result, expected, atol=1e-12)


class TestQUICKSchemeBuildKernel:
    def test_matches_compute(self):
        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=2, name="phi")
        _init_sin3d_cell(phi, geom)
        face_fluxes = _make_fluxes_level(box, dm, geom, ngrow=2)
        scheme = QUICK()
        for mfi in blockamr.MFIterator(phi.mf[0]):
            flux_data = _make_box_flux_data(face_fluxes, mfi, geom, scheme.stencil_width)
            kernel = scheme.build_kernel(flux_data)
            phi_arr = phi.mf[0].grown_array(mfi)
            result = kernel(phi_arr)
            phi_3d = phi_arr[:, :, :, 0]
            fx = flux_data.flux_x[:, :, :, 0]
            fy = flux_data.flux_y[:, :, :, 0]
            fz = flux_data.flux_z[:, :, :, 0]
            expected = scheme.compute(phi_3d, [fx, fy, fz], flux_data.dh)
            assert jnp.allclose(result, expected, atol=1e-12)


class TestLaplacianSchemeBuildKernel:
    def test_returns_callable(self):
        scheme = CentralDiffLaplacian()
        dh = jnp.array([1.0 / 32, 1.0 / 32, 1.0 / 32])
        gamma = jnp.ones((34, 34, 34))
        kernel = scheme.build_kernel(gamma, dh)
        assert callable(kernel)

    def test_matches_compute(self):
        scheme = CentralDiffLaplacian()
        field, *_ = _make_field()
        _init_sin3d(field)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        gamma = jnp.ones_like(phi)
        kernel = scheme.build_kernel(gamma, dh)
        result = kernel(phi)
        expected = scheme.compute(phi, gamma, dh)
        assert jnp.allclose(result, expected, atol=1e-12)


class TestGradSchemeBuildKernel:
    def test_returns_callable(self):
        scheme = CentralDiffGrad()
        dh = jnp.array([1.0 / 32, 1.0 / 32, 1.0 / 32])
        kernel = scheme.build_kernel(dh)
        assert callable(kernel)

    def test_matches_compute(self):
        scheme = CentralDiffGrad()
        field, *_ = _make_field()
        _init_sin3d(field)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        kernel = scheme.build_kernel(dh)
        result = kernel(phi)
        expected = scheme.compute(phi, dh)
        assert jnp.allclose(result, expected, atol=1e-12)



# ---------------------------------------------------------------------------
# Expression.__sub__ mutation test
# ---------------------------------------------------------------------------


class TestExpressionSubDoesNotMutate:
    def test_sub_preserves_original_coeff(self):
        """Expression.__sub__ must not mutate the operator's coeff."""
        from neon.blockamr.dsl.expression import Expression

        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        original_coeff = div_op.coeff

        expr = Expression()
        expr = expr - div_op

        assert div_op.coeff == original_coeff, (
            f"__sub__ mutated coeff from {original_coeff} to {div_op.coeff}"
        )


# ---------------------------------------------------------------------------
# Fused step test
# ---------------------------------------------------------------------------


class TestFusedStep:
    def test_fused_step_matches_sequential(self):
        """_fused_step with multiple functors matches sequential sum."""
        from neon.blockamr.dsl.solve import _fused_step

        mesh, box, dm, geom = _make_mesh()
        phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
        _init_sin3d_cell(phi, geom)
        ff = _make_fluxes(box, dm, geom)
        div_op = Div(ff, phi)
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), phi, coeff=-1.0)

        for mfi in blockamr.MFIterator(phi.mf[0]):
            phi_arr = jnp.asarray(phi.mf[0].grown_array(mfi))
            k_div = div_op.build_kernel(mfi, t=0.0)
            k_lap = lap_op.build_kernel(mfi, t=0.0)
            fused = _fused_step(phi_arr, [k_div, k_lap])
            sequential = k_div(phi_arr) + k_lap(phi_arr)
            assert jnp.allclose(fused, sequential, atol=1e-12)
