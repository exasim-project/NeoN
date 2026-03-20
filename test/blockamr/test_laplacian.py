# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import jax.numpy as jnp
import numpy as np
from blockamr.field import Field
from blockamr.operators.laplacian import Laplacian


def _make_field(n_cell=64, max_size=32, ngrow=1):
    """Create a periodic Field on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom)


def _init_sin3d(field):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = field.dx
    for mfi in blockamr.MFIterator(field.mf):
        arr = field.mf.host_array(mfi)
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


def _compute_laplacian_error(n_cell, gamma_func, analytical_func):
    """Compute max error of laplacian(gamma, phi) vs analytical on sin3d."""
    field = _make_field(n_cell=n_cell, max_size=n_cell, ngrow=1)
    _init_sin3d(field)

    lap_op = Laplacian(gamma_func, field)

    max_err = 0.0
    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = lap_op.build_kernel(mfi, t=0.0)
        result = kernel(phi)
        lo = mfi.valid_box().small_end()
        dx = field.geom.cell_size()
        prob_lo = field.geom.prob_lo()
        valid_arr = field.mf.host_array(mfi)
        nx, ny, nz = valid_arr.shape[:3]
        for i in range(nx):
            x = prob_lo[0] + (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = prob_lo[1] + (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = prob_lo[2] + (lo[2] + k + 0.5) * dx[2]
                    exact = analytical_func(x, y, z)
                    err = abs(float(result[i, j, k]) - exact)
                    if err > max_err:
                        max_err = err
    return max_err


def test_laplacian_const_gamma_convergence():
    """Laplacian with gamma=1 converges at O(dx^2) on sin3d.

    Analytical: nabla^2(sin3d) = -12*pi^2 * sin3d.
    """
    pi = math.pi

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    def analytical(x, y, z):
        return (
            -12.0 * pi**2
            * math.sin(2 * pi * x)
            * math.sin(2 * pi * y)
            * math.sin(2 * pi * z)
        )

    errors = []
    for n in [16, 32, 64]:
        err = _compute_laplacian_error(n, gamma_one, analytical)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 3.5, f"Ratio 16->32: {ratio_1:.2f}, expected ~4"
    assert ratio_2 > 3.5, f"Ratio 32->64: {ratio_2:.2f}, expected ~4"


def test_laplacian_variable_gamma_convergence():
    """Laplacian with variable gamma converges at O(dx^2).

    gamma(x) = 1 + 0.5*cos(2*pi*x)
    Analytical: div(gamma * grad(phi))
        = gamma * laplacian(phi) + grad(gamma) . grad(phi)
    """
    pi = math.pi

    def gamma_var(x, y, z, t):
        return 1.0 + 0.5 * np.cos(2 * pi * x)

    def analytical(x, y, z):
        s = lambda a: math.sin(2 * pi * a)  # noqa: E731
        c = lambda a: math.cos(2 * pi * a)  # noqa: E731
        phi = s(x) * s(y) * s(z)
        gamma = 1.0 + 0.5 * c(x)
        lap_phi = -12.0 * pi**2 * phi
        grad_gamma_dot_grad_phi = -pi * s(x) * 2 * pi * c(x) * s(y) * s(z)
        return gamma * lap_phi + grad_gamma_dot_grad_phi

    errors = []
    for n in [16, 32, 64]:
        err = _compute_laplacian_error(n, gamma_var, analytical)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 3.0, f"Ratio 16->32: {ratio_1:.2f}, expected ~4"
    assert ratio_2 > 3.0, f"Ratio 32->64: {ratio_2:.2f}, expected ~4"
