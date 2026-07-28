# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import blockamr
import jax.numpy as jnp
import numpy as np
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.operators.laplacian import Laplacian
from blockamr.flattened_boxes import flattened_boxes_from_mf, build_buckets
from blockamr.bucket_dispatch import process_bucket
from blockamr.cell_accessor import CellAccessor


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


def _compute_laplacian_error(n_cell, gamma_func, analytical_func):
    """Compute max error of laplacian(gamma, phi) vs analytical on sin3d.

    Uses the bucket dispatch pipeline to evaluate the laplacian kernel,
    then compares against the analytical solution.
    """
    mesh, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, geom)

    lap_op = Laplacian(gamma_func, phi)

    # Flatten and dispatch through buckets
    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in geom.cell_size())
    buckets = build_buckets(fb, dh, lev=0)

    max_err = 0.0
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    meta = mf.fab_metadata()

    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernel = lap_op.build_kernel(bucket, t=0.0)

        # Evaluate the laplacian via vmap (same as process_bucket but just the kernel)
        import jax

        def eval_one_box(box_idx):
            bound_k = kernel.for_box(bucket, box_idx)
            Nx = bucket.Nx_arr[box_idx]
            Ny = bucket.Ny_arr[box_idx]
            Nz = bucket.Nz_arr[box_idx]

            def eval_one_cell(cell_idx):
                phi_acc = CellAccessor(
                    bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                    Nx, Ny, Nz, bucket.ng,
                )
                return bound_k(phi_acc)

            return jax.vmap(eval_one_cell)(jnp.arange(bucket.n_cells_padded))

        result = jax.jit(jax.vmap(eval_one_box))(jnp.arange(bucket.max_boxes))

        # Compare each valid box's result to analytical
        ng = bucket.ng
        for bi, mf_idx in enumerate(bucket.box_indices[:bucket.n_valid]):
            Nx = int(bucket.Nx_arr[bi])
            Ny = int(bucket.Ny_arr[bi])
            Nz = int(bucket.Nz_arr[bi])
            vNx = Nx - 2 * ng
            vNy = Ny - 2 * ng
            vNz = Nz - 2 * ng
            actual_n_cells = vNx * vNy * vNz
            cell_data = result[bi, :actual_n_cells]
            lap_3d = cell_data.reshape(vNz, vNy, vNx).transpose(2, 1, 0)

            # Get box coordinates
            m = meta[mf_idx]
            bx_lo_0 = None
            box_idx_counter = 0
            for mfi in blockamr.MFIterator(mf):
                if box_idx_counter == mf_idx:
                    bx = mfi.valid_box()
                    bx_lo_0 = bx.small_end()
                    break
                box_idx_counter += 1

            lo = bx_lo_0
            xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(vNx)])
            ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(vNy)])
            zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(vNz)])
            X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
            exact = analytical_func(X, Y, Z)
            err = float(jnp.max(jnp.abs(lap_3d - exact)))
            max_err = max(max_err, err)
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
            * jnp.sin(2 * pi * x)
            * jnp.sin(2 * pi * y)
            * jnp.sin(2 * pi * z)
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
        s = lambda a: jnp.sin(2 * pi * a)  # noqa: E731
        c = lambda a: jnp.cos(2 * pi * a)  # noqa: E731
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
