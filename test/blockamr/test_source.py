# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import neon.blockamr as blockamr
import jax.numpy as jnp
from neon.blockamr.field import CellField
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.source import Source


def _make_mesh(n_cell=64, max_size=32):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _init_sin3d(phi, geom):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
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


def test_source_exact():
    """Source(coeff_func, phi) = coeff_func * phi at cell centers (no stencil)."""
    mesh, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, geom)

    def coeff_func(x, y, z, t):
        return x**2 + y

    source_op = Source(coeff_func, phi)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = jnp.asarray(phi.mf[0].grown_array(mfi)[:, :, :, 0])
        kernel = source_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        lo = mfi.valid_box().small_end()
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        valid_arr = phi.mf[0].copy_to_host(mfi)
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
