# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import numpy as np
from blockamr.field import Field
from blockamr.dsl import exp, solve
from blockamr.dsl.expression import Expression


def _make_field(n_cell=64, max_size=32, ngrow=1, name="phi"):
    """Create a periodic Field wrapping a MultiFab + Geometry."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom, name=name)


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


def test_ddt_plus_div_creates_expression():
    """ddt(phi) + div(vel, phi) creates an Expression with 1 temporal + 1 spatial op."""
    field = _make_field()

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    expr = exp.ddt(field) + exp.div(vel, field)
    assert isinstance(expr, Expression)
    assert len(expr.temporal_ops) == 1
    assert len(expr.spatial_ops) == 1


def test_scalar_mul_operator():
    """Scalar * operator sets the coefficient."""
    field = _make_field()

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    div_op = 2.0 * exp.div(vel, field)
    assert div_op.coeff == 2.0


def test_expression_subtraction():
    """ddt(phi) - div(vel, phi) negates the spatial op coefficient."""
    field = _make_field()

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    expr = exp.ddt(field) - exp.div(vel, field)
    assert isinstance(expr, Expression)
    assert len(expr.spatial_ops) == 1
    assert expr.spatial_ops[0].coeff == -1.0


def test_solve_constant_field_unchanged():
    """Solving ddt(phi) + div(U=0, phi) = 0 leaves a constant field unchanged."""
    field = _make_field(n_cell=64, max_size=32, ngrow=1)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
        arr[:, :, :, 0] = 5.0

    def zero_vel(x, y, z, t):
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    expr = exp.ddt(field) + exp.div(zero_vel, field)
    solve(expr, t=0.0, dt=0.01)

    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.array(mfi)
        assert np.allclose(arr[:, :, :, 0], 5.0)


def test_diffusion_single_step():
    """One forward-Euler step of ddt(phi) - laplacian(1, phi) = 0.

    Verify: phi_new = phi_old + dt * laplacian(phi_old).
    """
    n_cell = 32
    field = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=1)
    _init_sin3d(field)

    phi_old = {}
    for mfi in blockamr.MFIterator(field.mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        phi_old[tuple(lo)] = field.mf.array(mfi)[:, :, :, 0].copy()

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    dt = 1e-5
    expr = exp.ddt(field) - exp.laplacian(gamma_one, field)
    solve(expr, t=0.0, dt=dt)

    pi = math.pi
    decay = 1.0 + dt * (-12.0 * pi**2)

    for mfi in blockamr.MFIterator(field.mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        arr_new = field.mf.array(mfi)[:, :, :, 0]
        arr_old = phi_old[tuple(lo)]
        expected = arr_old * decay
        assert np.allclose(arr_new, expected, atol=1e-4), (
            f"Max diff: {np.abs(arr_new - expected).max()}"
        )
