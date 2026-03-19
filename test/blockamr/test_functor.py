# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TDD tests for NamedTuple functor pattern (build_kernel)."""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import blockamr
from blockamr.field import Field, FaceField
from blockamr.operators.div import Div, build_face_fluxes
from blockamr.operators.laplacian import Laplacian
from blockamr.operators.grad import Grad
from blockamr.operators.source import Source
from blockamr.schemes.div_schemes import Upwind, Linear, VanLeer, QUICK
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.grad_schemes import CentralDiffGrad


def _make_field(n_cell=32, max_size=32, ngrow=1, name="phi"):
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom, name=name, box=box, dm=dm, max_size=max_size), box, dm, geom


def _init_sin3d(field):
    dx = field.dx
    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
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
    field.fill_boundary()


def _make_fluxes(box, dm, geom, ngrow=1):
    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    return build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0)


# ---------------------------------------------------------------------------
# FaceField DistributionMapping test
# ---------------------------------------------------------------------------


class TestFaceFieldSharesDm:
    def test_cross_multifab_access_via_shared_mfiter(self):
        """FaceField must share caller's dm so MFIterator works across MultiFabs."""
        field, box, dm, geom = _make_field()
        ff = FaceField(box, dm, geom, ncomp=1, ngrow=1, max_size=32)
        # If dm is shared, iterating field.mf and accessing ff[d].mf with same mfi works
        for mfi in blockamr.MFIterator(field.mf):
            for d in range(3):
                arr = ff[d].mf.grown_array(mfi)
                assert arr.shape[0] > 0


# ---------------------------------------------------------------------------
# Operator build_kernel(mfi, t) tests
# ---------------------------------------------------------------------------


class TestDivBuildKernelMfi:
    def test_returns_callable(self):
        """Div.build_kernel(mfi, t) returns a callable."""
        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom)
        div_op = Div(face_fluxes, field)
        for mfi in blockamr.MFIterator(field.mf):
            kernel = div_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestLaplacianBuildKernelMfi:
    def test_returns_callable(self):
        field, *_ = _make_field()
        _init_sin3d(field)
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), field)
        for mfi in blockamr.MFIterator(field.mf):
            kernel = lap_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestGradBuildKernelMfi:
    def test_returns_callable(self):
        field, *_ = _make_field()
        grad_op = Grad(field)
        for mfi in blockamr.MFIterator(field.mf):
            kernel = grad_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestSourceBuildKernelMfi:
    def test_returns_callable(self):
        field, *_ = _make_field()
        src_op = Source(lambda x, y, z, t: np.ones_like(x), field)
        for mfi in blockamr.MFIterator(field.mf):
            kernel = src_op.build_kernel(mfi, t=0.0)
            assert callable(kernel)


class TestFusedStepMfi:
    def test_fused_step_with_mfi(self):
        """_fused_step with mfi-based kernels matches sequential sum."""
        from blockamr.dsl.solve import _fused_step

        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom)
        div_op = Div(face_fluxes, field)
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), field, coeff=-1.0)

        for mfi in blockamr.MFIterator(field.mf):
            phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
            k_div = div_op.build_kernel(mfi, t=0.0)
            k_lap = lap_op.build_kernel(mfi, t=0.0)
            fused = _fused_step(phi, [k_div, k_lap])
            sequential = k_div(phi) + k_lap(phi)
            assert jnp.allclose(fused, sequential, atol=1e-12)


# ---------------------------------------------------------------------------
# Scheme-level build_kernel tests
# ---------------------------------------------------------------------------


def _extract_fluxes(face_fluxes, patch, stencil_width):
    """Extract flux arrays for a patch from a FaceField."""
    w = stencil_width
    box_lo = tuple(patch.box.small_end())
    fluxes = []
    for dim in range(3):
        for fp in face_fluxes[dim].patches():
            fp_lo = tuple(fp.box.small_end())
            if fp_lo == box_lo:
                flux = jnp.array(fp.grown_arr[:, :, :, 0])
                sl = [slice(None)] * 3
                sl[dim] = slice(w, -w) if w > 0 else slice(None)
                fluxes.append(flux[tuple(sl)])
                break
    return fluxes


class TestUpwindSchemeBuildKernel:
    def test_returns_named_tuple(self):
        """Upwind.build_kernel() returns a NamedTuple."""
        scheme = Upwind()
        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=1)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        assert isinstance(kernel, tuple)
        assert hasattr(kernel, '_fields')  # NamedTuple check

    def test_callable(self):
        """Upwind kernel functor is callable with phi."""
        scheme = Upwind()
        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=1)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        result = kernel(phi)
        assert result.shape == patch.valid_arr.shape[:3]

    def test_matches_compute(self):
        """Upwind build_kernel() output matches compute()."""
        scheme = Upwind()
        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=1)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        result = kernel(phi)
        expected = scheme.compute(phi, fluxes, dh)
        assert jnp.allclose(result, expected, atol=1e-12)


class TestLinearSchemeBuildKernel:
    def test_matches_compute(self):
        scheme = Linear()
        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=1)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        result = kernel(phi)
        expected = scheme.compute(phi, fluxes, dh)
        assert jnp.allclose(result, expected, atol=1e-12)


class TestVanLeerSchemeBuildKernel:
    def test_matches_compute(self):
        scheme = VanLeer()
        field, box, dm, geom = _make_field(ngrow=2)
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=2)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        result = kernel(phi)
        expected = scheme.compute(phi, fluxes, dh)
        assert jnp.allclose(result, expected, atol=1e-12)


class TestQUICKSchemeBuildKernel:
    def test_matches_compute(self):
        scheme = QUICK()
        field, box, dm, geom = _make_field(ngrow=2)
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom, ngrow=2)
        patch = next(field.patches())
        dh = jnp.array(patch.geom.cell_size())
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])
        fluxes = _extract_fluxes(face_fluxes, patch, scheme.stencil_width)
        kernel = scheme.build_kernel(fluxes, dh)
        result = kernel(phi)
        expected = scheme.compute(phi, fluxes, dh)
        assert jnp.allclose(result, expected, atol=1e-12)


class TestLaplacianSchemeBuildKernel:
    def test_returns_named_tuple(self):
        scheme = CentralDiffLaplacian()
        dh = jnp.array([1.0 / 32, 1.0 / 32, 1.0 / 32])
        gamma = jnp.ones((34, 34, 34))
        kernel = scheme.build_kernel(gamma, dh)
        assert isinstance(kernel, tuple)
        assert hasattr(kernel, '_fields')

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
    def test_returns_named_tuple(self):
        scheme = CentralDiffGrad()
        dh = jnp.array([1.0 / 32, 1.0 / 32, 1.0 / 32])
        kernel = scheme.build_kernel(dh)
        assert isinstance(kernel, tuple)
        assert hasattr(kernel, '_fields')

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
        from blockamr.dsl.expression import Expression

        field, box, dm, geom = _make_field()
        face_fluxes = _make_fluxes(box, dm, geom)
        div_op = Div(face_fluxes, field)
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
        from blockamr.dsl.solve import _fused_step

        field, box, dm, geom = _make_field()
        _init_sin3d(field)
        face_fluxes = _make_fluxes(box, dm, geom)
        div_op = Div(face_fluxes, field)
        lap_op = Laplacian(lambda x, y, z, t: np.ones_like(x), field, coeff=-1.0)

        for mfi in blockamr.MFIterator(field.mf):
            phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
            k_div = div_op.build_kernel(mfi, t=0.0)
            k_lap = lap_op.build_kernel(mfi, t=0.0)
            fused = _fused_step(phi, [k_div, k_lap])
            sequential = k_div(phi) + k_lap(phi)
            assert jnp.allclose(fused, sequential, atol=1e-12)
