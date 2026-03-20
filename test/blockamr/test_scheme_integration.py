# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import jax.numpy as jnp
import numpy as np
from blockamr.field import Field
from blockamr.dsl import exp, solve
from blockamr.operators.div import Div, build_face_fluxes
from blockamr.operators.grad import Grad
from blockamr.operators.laplacian import Laplacian
from blockamr.schemes.div_schemes import QUICK, Linear, Upwind, VanLeer
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian


def _make_field(n_cell=64, max_size=32, ngrow=1):
    """Create a periodic Field on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom, box=box, dm=dm, max_size=max_size), box, dm, geom


def _uniform_vel(x, y, z, t):
    return np.ones_like(x), np.ones_like(x), np.ones_like(x)


def _x_vel(x, y, z, t):
    return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)


def test_div_default_unchanged():
    """Div without explicit scheme matches old (Upwind) behaviour."""
    field, box, dm, geom = _make_field(n_cell=64, max_size=32, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        arr[:, :, :, 0] = 1.0
    field.fill_boundary()

    face_fluxes = build_face_fluxes(_uniform_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(face_fluxes, field)
    assert isinstance(div_op.scheme, Upwind)

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        assert np.allclose(result, 0.0, atol=1e-12)


def test_div_with_linear_scheme():
    """Div with Linear scheme gives different (2nd-order) results vs Upwind."""
    n_cell = 32
    field, box, dm, geom = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=1)
    dx = field.dx

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
    field.fill_boundary()

    face_fluxes_up = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    face_fluxes_lin = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_upwind = Div(face_fluxes_up, field)
    div_linear = Div(face_fluxes_lin, field, scheme=Linear())

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        result_upwind = div_upwind.build_kernel(mfi, t=0.0)(phi)
        result_linear = div_linear.build_kernel(mfi, t=0.0)(phi)
        # They should differ (Linear is 2nd-order, Upwind is 1st-order)
        assert not np.allclose(result_upwind, result_linear, atol=1e-6)


def test_laplacian_default_scheme():
    """Laplacian defaults to CentralDiffLaplacian."""
    field, *_ = _make_field(n_cell=32, max_size=32, ngrow=1)

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    lap_op = Laplacian(gamma_one, field)
    assert isinstance(lap_op.scheme, CentralDiffLaplacian)


def test_grad_default_scheme():
    """Grad defaults to CentralDiffGrad."""
    field, *_ = _make_field(n_cell=32, max_size=32, ngrow=1)
    grad_op = Grad(field)
    assert isinstance(grad_op.scheme, CentralDiffGrad)


def test_solve_with_schemes_dict():
    """solve() with schemes dict overrides operator's default scheme."""
    field, box, dm, geom = _make_field(n_cell=32, max_size=32, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        dx = field.dx
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
    field.fill_boundary()
    phi_before = {}
    for mfi in blockamr.MFIterator(field.mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        phi_before[tuple(lo)] = field.mf.host_array(mfi)[:, :, :, 0].copy()

    face_fluxes = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    expr = exp.ddt(field) + exp.div(face_fluxes, field)
    solve(expr, t=0.0, dt=1e-4, schemes={"Div": Linear()})

    # The field should have changed (not zero div for non-uniform field)
    for mfi in blockamr.MFIterator(field.mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        arr_new = field.mf.host_array(mfi)[:, :, :, 0]
        assert not np.allclose(arr_new, phi_before[tuple(lo)])


def test_div_with_vanleer_scheme():
    """Div with VanLeer scheme runs without error (requires wider stencil, ngrow>=2)."""
    n_cell = 32
    # VanLeer needs 2 ghost cells
    field, box, dm, geom = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=2)
    dx = field.dx

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
    field.fill_boundary()

    face_fluxes = build_face_fluxes(_x_vel, box, dm, geom, ngrow=2, t=0.0)
    div_vanleer = Div(face_fluxes, field, scheme=VanLeer())

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = div_vanleer.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        # Should produce finite results
        assert np.all(np.isfinite(result))


def test_factory_div_with_scheme():
    """exp.div() accepts optional scheme parameter."""
    field, box, dm, geom = _make_field(n_cell=32, max_size=32, ngrow=1)
    face_fluxes = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = exp.div(face_fluxes, field, scheme=Linear())
    assert isinstance(div_op.scheme, Linear)


def test_solve_backward_compat():
    """solve() without schemes kwarg works as before."""
    field, box, dm, geom = _make_field(n_cell=64, max_size=32, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        arr[:, :, :, 0] = 5.0

    def zero_vel(x, y, z, t):
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    face_fluxes = build_face_fluxes(zero_vel, box, dm, geom, ngrow=1, t=0.0)
    expr = exp.ddt(field) + exp.div(face_fluxes, field)
    solve(expr, t=0.0, dt=0.01)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
        assert np.allclose(arr[:, :, :, 0], 5.0)
