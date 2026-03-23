# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import neon.blockamr as blockamr
import jax.numpy as jnp
from neon.blockamr.field import CellField
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.grad import Grad


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
    pi = math.pi
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        arr[:, :, :, 0] = jnp.sin(2 * pi * X) * jnp.sin(2 * pi * Y) * jnp.sin(2 * pi * Z)
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)


def _compute_grad_error(n_cell):
    """Compute max error of grad(sin3d) vs analytical."""
    mesh, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, geom)

    grad_op = Grad(phi)
    pi = math.pi

    max_err = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = jnp.asarray(phi.mf[0].grown_array(mfi)[:, :, :, 0])
        kernel = grad_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        lo = mfi.valid_box().small_end()
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        valid_arr = phi.mf[0].copy_to_host(mfi)
        nx, ny, nz = valid_arr.shape[:3]
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        sx, sy, sz = jnp.sin(2 * pi * X), jnp.sin(2 * pi * Y), jnp.sin(2 * pi * Z)
        cx, cy, cz = jnp.cos(2 * pi * X), jnp.cos(2 * pi * Y), jnp.cos(2 * pi * Z)
        exact = jnp.stack([2 * pi * cx * sy * sz,
                           2 * pi * sx * cy * sz,
                           2 * pi * sx * sy * cz], axis=-1)
        err = float(jnp.max(jnp.abs(result - exact)))
        max_err = max(max_err, err)
    return max_err


def test_gradient_convergence():
    """Central-difference gradient converges at O(dx^2) on sin3d."""
    errors = []
    for n in [16, 32, 64]:
        err = _compute_grad_error(n)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 3.5, f"Ratio 16->32: {ratio_1:.2f}, expected ~4"
    assert ratio_2 > 3.5, f"Ratio 32->64: {ratio_2:.2f}, expected ~4"
