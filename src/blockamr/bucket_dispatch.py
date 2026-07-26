# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Bucketed dispatch: nested vmap over boxes and cells,
plus flat element-level dispatch with lax.fori_loop chunks."""

import jax
import jax.numpy as jnp

from .cell_accessor import CellAccessor


@jax.jit
def process_bucket(bucket, dt_over_coeff, kernels):
    """Process all boxes in one bucket with nested vmap.

    Args:
        bucket: BucketContext (eqx.Module) — static tier constants, traced
                per-box geometry and data.
        dt_over_coeff: scalar — dt / ddt_coeff.
        kernels: tuple of eqx.Module kernels with for_box(), __call__(phi),
                 and ncomp (static field).

    Returns:
        ncomp=1: (max_boxes, n_cells_padded)
        ncomp>1: (max_boxes, n_cells_padded, ncomp)
    """
    ncomp = kernels[0].ncomp  # static → compile-time constant

    def process_one_box(box_idx):
        Nx = bucket.Nx_arr[box_idx]
        Ny = bucket.Ny_arr[box_idx]
        Nz = bucket.Nz_arr[box_idx]
        actual_n_cells = bucket.n_cells_arr[box_idx]
        bound_kernels = tuple(k.for_box(bucket, box_idx) for k in kernels)

        def process_one_cell(cell_idx):
            if ncomp == 1:
                phi = CellAccessor(
                    bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                    Nx, Ny, Nz, bucket.ng,
                )
                total = 0.0
                for k in bound_kernels:
                    total = total + k(phi)
                result = phi.center - dt_over_coeff * total
                return jnp.where(cell_idx < actual_n_cells, result, 0.0)
            else:
                results = []
                for comp in range(ncomp):
                    phi = CellAccessor(
                        bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                        Nx, Ny, Nz, bucket.ng,
                        nc=ncomp, comp=comp,
                    )
                    total = 0.0
                    for k in bound_kernels:
                        total = total + k(phi)
                    results.append(phi.center - dt_over_coeff * total)
                result = jnp.array(results)
                return jnp.where(cell_idx < actual_n_cells, result, 0.0)

        return jax.vmap(process_one_cell)(jnp.arange(bucket.n_cells_padded))

    return jax.vmap(process_one_box)(jnp.arange(bucket.max_boxes))


@jax.jit
def evaluate_bucket(bucket, kernels):
    """Evaluate spatial operators for all boxes — return source term only.

    Unlike process_bucket, does NOT do the time integration step
    (phi - dt * source). Just returns the sum of kernel contributions.

    Returns:
        ncomp=1: (max_boxes, n_cells_padded)
        ncomp>1: (max_boxes, n_cells_padded, ncomp)
    """
    ncomp = kernels[0].ncomp

    def eval_one_box(box_idx):
        Nx = bucket.Nx_arr[box_idx]
        Ny = bucket.Ny_arr[box_idx]
        Nz = bucket.Nz_arr[box_idx]
        actual_n_cells = bucket.n_cells_arr[box_idx]
        bound_kernels = tuple(k.for_box(bucket, box_idx) for k in kernels)

        def eval_one_cell(cell_idx):
            if ncomp == 1:
                phi = CellAccessor(
                    bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                    Nx, Ny, Nz, bucket.ng,
                )
                total = 0.0
                for k in bound_kernels:
                    total = total + k(phi)
                return jnp.where(cell_idx < actual_n_cells, total, 0.0)
            else:
                results = []
                for comp in range(ncomp):
                    phi = CellAccessor(
                        bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                        Nx, Ny, Nz, bucket.ng,
                        nc=ncomp, comp=comp,
                    )
                    total = 0.0
                    for k in bound_kernels:
                        total = total + k(phi)
                    results.append(total)
                result = jnp.array(results)
                return jnp.where(cell_idx < actual_n_cells, result, 0.0)

        return jax.vmap(eval_one_cell)(jnp.arange(bucket.n_cells_padded))

    return jax.vmap(eval_one_box)(jnp.arange(bucket.max_boxes))


# ---------------------------------------------------------------------------
# Flat element-level dispatch with lax.fori_loop chunks
# ---------------------------------------------------------------------------

@jax.jit
def process_flat(bucket, elem_map, dt_over_coeff, kernels):
    """Process all valid cells via flat element-level dispatch.

    One JIT'd kernel launch processes all boxes on a level
    using a single vmap over all valid cells.

    Args:
        bucket: BucketContext covering all boxes on the level.
        elem_map: ElementMap with flat element → (box, cell) mapping.
        dt_over_coeff: scalar — dt / ddt_coeff.
        kernels: tuple of eqx.Module kernels with for_box() and __call__().

    Returns:
        ncomp=1: (total_padded,) — use [:total_valid] for real data.
        ncomp>1: (total_padded, ncomp).
    """
    ncomp = kernels[0].ncomp

    def _process_one(flat_idx):
        box_idx = elem_map.elem_to_box[flat_idx]
        cell_idx = elem_map.elem_to_cell_idx[flat_idx]
        is_valid = flat_idx < elem_map.total_valid
        bound_kernels = tuple(k.for_box(bucket, box_idx) for k in kernels)

        if ncomp == 1:
            phi = CellAccessor(
                bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
                bucket.Nz_arr[box_idx], bucket.ng,
            )
            total = 0.0
            for k in bound_kernels:
                total = total + k(phi)
            result = phi.center - dt_over_coeff * total
            return jnp.where(is_valid, result, 0.0)
        else:
            results = []
            for comp in range(ncomp):
                phi = CellAccessor(
                    bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                    bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
                    bucket.Nz_arr[box_idx], bucket.ng,
                    nc=ncomp, comp=comp,
                )
                total = 0.0
                for k in bound_kernels:
                    total = total + k(phi)
                results.append(phi.center - dt_over_coeff * total)
            result = jnp.array(results)
            return jnp.where(is_valid, result, 0.0)

    return jax.vmap(_process_one)(jnp.arange(elem_map.total_padded))


@jax.jit
def evaluate_flat(bucket, elem_map, kernels):
    """Evaluate spatial operators via flat dispatch — source term only.

    Same structure as process_flat but returns the sum of kernel
    contributions without the time-integration step.
    """
    ncomp = kernels[0].ncomp

    def _eval_one(flat_idx):
        box_idx = elem_map.elem_to_box[flat_idx]
        cell_idx = elem_map.elem_to_cell_idx[flat_idx]
        is_valid = flat_idx < elem_map.total_valid
        bound_kernels = tuple(k.for_box(bucket, box_idx) for k in kernels)

        if ncomp == 1:
            phi = CellAccessor(
                bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
                bucket.Nz_arr[box_idx], bucket.ng,
            )
            total = 0.0
            for k in bound_kernels:
                total = total + k(phi)
            return jnp.where(is_valid, total, 0.0)
        else:
            results = []
            for comp in range(ncomp):
                phi = CellAccessor(
                    bucket.cell_buf, bucket.box_offsets[box_idx], cell_idx,
                    bucket.Nx_arr[box_idx], bucket.Ny_arr[box_idx],
                    bucket.Nz_arr[box_idx], bucket.ng,
                    nc=ncomp, comp=comp,
                )
                total = 0.0
                for k in bound_kernels:
                    total = total + k(phi)
                results.append(total)
            result = jnp.array(results)
            return jnp.where(is_valid, result, 0.0)

    return jax.vmap(_eval_one)(jnp.arange(elem_map.total_padded))
