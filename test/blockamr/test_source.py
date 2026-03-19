# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import jax.numpy as jnp
from blockamr.field import Field
from blockamr.operators.source import Source


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


def test_source_exact():
    """Source(coeff_func, phi) = coeff_func * phi at cell centers (no stencil)."""
    field = _make_field(n_cell=32, max_size=32, ngrow=1)
    _init_sin3d(field)

    def coeff_func(x, y, z, t):
        return x**2 + y

    source_op = Source(coeff_func, field)

    for mfi in blockamr.MFIterator(field.mf):
        phi = jnp.asarray(field.mf.grown_array(mfi)[:, :, :, 0])
        kernel = source_op.build_kernel(mfi, t=0.0)
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
                    phi_val = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
                    exact = (x**2 + y) * phi_val
                    assert abs(float(result[i, j, k]) - exact) < 1e-14, (
                        f"At ({x:.3f},{y:.3f},{z:.3f}): "
                        f"got {float(result[i, j, k])}, expected {exact}"
                    )
