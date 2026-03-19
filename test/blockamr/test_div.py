# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import jax.numpy as jnp
import numpy as np
from blockamr.field import Field
from blockamr.operators.div import Div, build_face_fluxes


def _make_field(n_cell=64, max_size=32, ngrow=1):
    """Create a periodic Field on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom), box, dm, geom


def _init_sin3d(field):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
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


def test_div_uniform_field_is_zero():
    """Divergence of a uniform field should be zero."""
    field, box, dm, geom = _make_field(n_cell=64, max_size=32, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
        arr[:, :, :, 0] = 1.0
    field.fill_boundary()

    def uniform_vel(x, y, z, t):
        u = np.ones_like(x)
        v = np.ones_like(x)
        w = np.ones_like(x)
        return u, v, w

    face_fluxes = build_face_fluxes(uniform_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(face_fluxes, field)

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        assert np.allclose(result, 0.0, atol=1e-12), f"max div = {np.abs(result).max()}"


def test_div_sin_field():
    """Divergence of sin(2*pi*x) with u=1 approximates 2*pi*cos(2*pi*x)."""
    field, box, dm, geom = _make_field(n_cell=64, max_size=32, ngrow=1)
    dx = field.dx

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
    field.fill_boundary()

    def x_vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    face_fluxes = build_face_fluxes(x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(face_fluxes, field)

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        lo = mfi.valid_box().small_end()
        nx = result.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            analytic = 2 * math.pi * math.cos(2 * math.pi * x)
            assert abs(result[i, 0, 0] - analytic) < 0.6, (
                f"At x={x:.3f}: got {result[i, 0, 0]:.4f}, expected {analytic:.4f}"
            )


def _compute_div_error(n_cell):
    """Compute max error of div(U*phi) with U=(1,0,0) vs analytical on sin3d."""
    field, box, dm, geom = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=1)
    _init_sin3d(field)

    def x_vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    face_fluxes = build_face_fluxes(x_vel, box, dm, geom, ngrow=1, t=0.0, max_size=n_cell)
    div_op = Div(face_fluxes, field)
    pi = math.pi

    max_err = 0.0
    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        lo = mfi.valid_box().small_end()
        dx = field.geom.cell_size()
        prob_lo = field.geom.prob_lo()
        valid_arr = field.mf.array(mfi)
        nx, ny, nz = valid_arr.shape[:3]
        for i in range(nx):
            x = prob_lo[0] + (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = prob_lo[1] + (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = prob_lo[2] + (lo[2] + k + 0.5) * dx[2]
                    exact = (
                        2 * pi
                        * math.cos(2 * pi * x)
                        * math.sin(2 * pi * y)
                        * math.sin(2 * pi * z)
                    )
                    err = abs(float(result[i, j, k]) - exact)
                    if err > max_err:
                        max_err = err
    return max_err


def test_div_sin_convergence():
    """First-order upwind div converges at O(dx) on sin3d with U=(1,0,0).

    Analytical: div(U*phi) = dphi/dx = 2*pi*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*z).
    Error ratio should be ~2 (first-order).
    """
    errors = []
    for n in [16, 32, 64]:
        err = _compute_div_error(n)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 1.8, f"Ratio 16->32: {ratio_1:.2f}, expected ~2"
    assert ratio_2 > 1.8, f"Ratio 32->64: {ratio_2:.2f}, expected ~2"
