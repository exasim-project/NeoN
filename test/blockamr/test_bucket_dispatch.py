# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for bucketed dispatch and recompilation behavior."""

import jax
import jax.numpy as jnp

from neon.blockamr.flattened_boxes import FlattenedBoxes, BucketContext, build_buckets
from neon.blockamr.cell_kernels import CellLaplacianKernel
from neon.blockamr.bucket_dispatch import process_bucket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quad_box(Nx, Ny, Nz, dx=1.0):
    """Fill box with u = (i*dx)^2 + (j*dx)^2 + (k*dx)^2 in Fortran order."""
    buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                buf = buf.at[i + Nx * j + Nx * Ny * k].set(
                    (i * dx) ** 2 + (j * dx) ** 2 + (k * dx) ** 2
                )
    return buf


def _make_bucket(cell_buf, offsets_list, Nx, Ny, Nz, ng, dh=(1.0, 1.0, 1.0),
                 n_pad=None):
    """Create a BucketContext for uniform-shape boxes."""
    from neon.blockamr.flattened_boxes import _cell_tier
    n = len(offsets_list)
    pad = n_pad or n
    dummy_off = offsets_list[0] if offsets_list else 0
    padded_offsets = list(offsets_list) + [dummy_off] * (pad - n)

    vNx = Nx - 2 * ng
    vNy = Ny - 2 * ng
    vNz = Nz - 2 * ng
    n_cells = vNx * vNy * vNz

    return BucketContext(
        box_offsets=jnp.array(padded_offsets[:pad], dtype=jnp.int32),
        cell_buf=cell_buf,
        Nx_arr=jnp.array([Nx] * pad, dtype=jnp.int32),
        Ny_arr=jnp.array([Ny] * pad, dtype=jnp.int32),
        Nz_arr=jnp.array([Nz] * pad, dtype=jnp.int32),
        n_cells_arr=jnp.array([n_cells] * pad, dtype=jnp.int32),
        dh_arr=jnp.array([list(dh)] * pad, dtype=jnp.float64),
        ng=ng,
        n_cells_padded=_cell_tier(n_cells),
        max_boxes=pad,
        n_valid=n,
        box_indices=tuple(range(n)),
    )


def _n_cells(Nx, Ny, Nz, ng):
    return (Nx - 2 * ng) * (Ny - 2 * ng) * (Nz - 2 * ng)


# ---------------------------------------------------------------------------
# Pytree
# ---------------------------------------------------------------------------

def test_flattened_boxes_is_jax_pytree():
    buf = jnp.arange(10, dtype=jnp.float64)
    fb = FlattenedBoxes(
        contiguous_array=buf,
        offsets=jnp.array([0, 5], dtype=jnp.int32),
        shapes=((3, 2, 1, 1), (5, 2, 1, 1)),
        n_grow=0,
    )
    # shapes and n_grow are static, so only contiguous_array and offsets are leaves
    assert len(jax.tree.leaves(fb)) == 2

    @jax.jit
    def read_first(fb):
        return fb.contiguous_array[0]
    assert float(read_first(fb)) == 0.0


def test_bucket_context_is_jax_pytree():
    buf = jnp.zeros(100)
    bucket = _make_bucket(buf, [0, 50], 6, 6, 4, 1, n_pad=4)
    # Traced leaves: box_offsets, cell_buf, Nx_arr, Ny_arr, Nz_arr,
    #                n_cells_arr, dh_arr
    assert len(jax.tree.leaves(bucket)) == 7


# ---------------------------------------------------------------------------
# process_bucket — correctness
# ---------------------------------------------------------------------------

def test_process_bucket_laplacian():
    from neon.blockamr.flattened_boxes import _cell_tier
    Nx, Ny, Nz, ng = 6, 6, 4, 1
    n_pad = 4
    box_size = Nx * Ny * Nz
    nc = _n_cells(Nx, Ny, Nz, ng)
    nc_padded = _cell_tier(nc)

    buf = jnp.concatenate(
        [_make_quad_box(Nx, Ny, Nz)] * 2 + [jnp.zeros(box_size)] * (n_pad - 2)
    )
    bucket = _make_bucket(buf, [0, box_size], Nx, Ny, Nz, ng, n_pad=n_pad)
    # dh is now a per-box array — kernel gets it via for_box()
    dh_arr = jnp.array([[1.0, 1.0, 1.0]] * n_pad, dtype=jnp.float64)
    kernel = CellLaplacianKernel(dh=dh_arr[0], coeff=1.0)

    result = process_bucket(bucket, 1.0, (kernel,))
    assert result.shape == (n_pad, nc_padded)
    # center(0) = (1^2+1^2+1^2) = 3.0, lap = 6.0
    # result = center - dt * lap = 3.0 - 1.0 * 6.0 = -3.0
    assert abs(float(result[0, 0]) - (-3.0)) < 1e-10
    assert abs(float(result[1, 0]) - (-3.0)) < 1e-10


# ---------------------------------------------------------------------------
# Recompilation tests
# ---------------------------------------------------------------------------

def test_no_recompile_same_shapes():
    """Changing offsets/data with same static fields → 0 compiles."""
    Nx, Ny, Nz, ng = 6, 6, 4, 1
    n_pad = 4
    box_size = Nx * Ny * Nz
    dh_arr = jnp.array([[1.0, 1.0, 1.0]] * n_pad, dtype=jnp.float64)
    kernel = CellLaplacianKernel(dh=dh_arr[0], coeff=1.0)

    box = _make_quad_box(Nx, Ny, Nz)
    buf = jnp.concatenate([box, box] + [jnp.zeros(box_size)] * (n_pad - 2))
    bucket = _make_bucket(buf, [0, box_size], Nx, Ny, Nz, ng, n_pad=n_pad)

    _ = process_bucket(bucket, 1.0, (kernel,))
    jax.block_until_ready(_)

    compiles = [0]
    def count(ev, dur, **kw):
        if ev == "/jax/core/compile/backend_compile_duration":
            compiles[0] += 1
    jax.monitoring.register_event_duration_secs_listener(count)

    box2 = _make_quad_box(Nx, Ny, Nz, dx=2.0)
    buf2 = jnp.concatenate(
        [box2, box2] + [jnp.zeros(box_size)] * (n_pad - 2)
    )
    bucket2 = _make_bucket(buf2, [0, box_size], Nx, Ny, Nz, ng,
                           n_pad=n_pad)
    compiles[0] = 0
    _ = process_bucket(bucket2, 1.0, (kernel,))
    jax.block_until_ready(_)
    jax.monitoring.unregister_event_duration_listener(count)
    assert compiles[0] == 0


def test_variable_max_boxes_recompiles():
    """Changing max_boxes (static) triggers recompilation."""
    Nx, Ny, Nz, ng = 6, 6, 4, 1
    box_size = Nx * Ny * Nz
    dh_arr1 = jnp.array([[1.0, 1.0, 1.0]] * 1, dtype=jnp.float64)
    kernel = CellLaplacianKernel(dh=dh_arr1[0], coeff=1.0)

    buf_a = _make_quad_box(Nx, Ny, Nz)
    bucket_a = _make_bucket(buf_a, [0], Nx, Ny, Nz, ng)
    _ = process_bucket(bucket_a, 0.0, (kernel,))
    jax.block_until_ready(_)

    compiles = [0]
    def count(ev, dur, **kw):
        if ev == "/jax/core/compile/backend_compile_duration":
            compiles[0] += 1
    jax.monitoring.register_event_duration_secs_listener(count)

    buf_b = jnp.concatenate([buf_a, buf_a])
    bucket_b = _make_bucket(buf_b, [0, box_size], Nx, Ny, Nz, ng)
    _ = process_bucket(bucket_b, 0.0, (kernel,))
    jax.block_until_ready(_)
    jax.monitoring.unregister_event_duration_listener(count)
    assert compiles[0] > 0


# ---------------------------------------------------------------------------
# build_buckets
# ---------------------------------------------------------------------------

def test_build_buckets_groups_by_cell_tier():
    """Boxes with different shapes but same cell-count tier go in one bucket."""
    box_a_size = 6 * 6 * 4   # valid cells (ng=1): 4*4*2 = 32
    box_b_size = 10 * 10 * 4  # valid cells (ng=1): 8*8*2 = 128
    buf = jnp.zeros(2 * box_a_size + box_b_size)
    fb = FlattenedBoxes(
        contiguous_array=buf,
        offsets=jnp.array([0, box_a_size, 2 * box_a_size], dtype=jnp.int32),
        shapes=((6, 6, 4, 1), (6, 6, 4, 1), (10, 10, 4, 1)),
        n_grow=1,
    )
    buckets = build_buckets(fb, dh=(1.0, 1.0, 1.0))
    # tier 32 (2 boxes) and tier 128 (1 box) → 2 buckets
    assert len(buckets) == 2

    # Find the bucket with 2 valid boxes (the 6x6x4 ones)
    b_small = [b for b in buckets if b.n_valid == 2][0]
    assert b_small.n_cells_padded == 32
    assert b_small.max_boxes == 2  # power-of-2 padded


def test_build_buckets_pads_to_power_of_2():
    """max_boxes is power-of-2 padded from n_valid."""
    box_size = 6 * 6 * 4
    fb = FlattenedBoxes(
        contiguous_array=jnp.zeros(5 * box_size),
        offsets=jnp.array([i * box_size for i in range(5)], dtype=jnp.int32),
        shapes=((6, 6, 4, 1),) * 5,
        n_grow=1,
    )
    buckets = build_buckets(fb, dh=(1.0, 1.0, 1.0))
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.n_valid == 5
    assert bucket.max_boxes == 8  # next power of 2
    assert bucket.box_offsets.shape == (8,)


def test_build_buckets_same_cell_count_different_shapes():
    """Boxes (6,4,6) and (4,6,6) have same cell count → same bucket."""
    # (6,4,6) with ng=1: valid = 4*2*4 = 32
    # (4,6,6) with ng=1: valid = 2*4*4 = 32
    box_a_size = 6 * 4 * 6
    box_b_size = 4 * 6 * 6
    buf = jnp.zeros(box_a_size + box_b_size)
    fb = FlattenedBoxes(
        contiguous_array=buf,
        offsets=jnp.array([0, box_a_size], dtype=jnp.int32),
        shapes=((6, 4, 6, 1), (4, 6, 6, 1)),
        n_grow=1,
    )
    buckets = build_buckets(fb, dh=(1.0, 1.0, 1.0))
    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.n_valid == 2
    # Per-box Nx differs
    assert int(bucket.Nx_arr[0]) == 6
    assert int(bucket.Nx_arr[1]) == 4


def test_build_buckets_single_box_minimal_padding():
    """Single box gets max_boxes=1, not a large constant."""
    box_size = 6 * 6 * 4
    fb = FlattenedBoxes(
        contiguous_array=jnp.zeros(box_size),
        offsets=jnp.array([0], dtype=jnp.int32),
        shapes=((6, 6, 4, 1),),
        n_grow=1,
    )
    buckets = build_buckets(fb, dh=(1.0, 1.0, 1.0))
    assert len(buckets) == 1
    assert buckets[0].max_boxes == 1
    assert buckets[0].n_valid == 1


# ---------------------------------------------------------------------------
# Step 1: Verify CellAccessor works with traced Nx, Ny, Nz
# ---------------------------------------------------------------------------

def test_cell_accessor_traced_dims():
    """CellAccessor works when Nx, Ny, Nz are traced (not static) values.

    This is the foundation for Plan D: per-box traced geometry arrays.
    If JAX can't trace through the modular arithmetic (cell_idx % vNx with
    traced vNx), the entire approach is blocked.
    """
    from neon.blockamr.cell_accessor import CellAccessor

    Nx, Ny, Nz, ng = 6, 6, 4, 1
    box = _make_quad_box(Nx, Ny, Nz)  # u = i^2 + j^2 + k^2

    # JIT function where Nx, Ny, Nz are traced arguments (not static)
    @jax.jit
    def read_stencil(buf, Nx, Ny, Nz):
        cell_idx = 0  # first valid cell: i=1, j=1, k=1
        phi = CellAccessor(buf, 0, cell_idx, Nx, Ny, Nz, ng)
        return jnp.array([
            phi.center,
            phi.x[1], phi.x[-1],
            phi.y[1], phi.y[-1],
            phi.z[1], phi.z[-1],
        ])

    result = read_stencil(box, jnp.int32(Nx), jnp.int32(Ny), jnp.int32(Nz))
    # cell (1,1,1): center = 1+1+1 = 3
    assert abs(float(result[0]) - 3.0) < 1e-10
    # x+1 = (2,1,1) = 4+1+1 = 6
    assert abs(float(result[1]) - 6.0) < 1e-10
    # x-1 = (0,1,1) = 0+1+1 = 2
    assert abs(float(result[2]) - 2.0) < 1e-10
    # y+1 = (1,2,1) = 1+4+1 = 6
    assert abs(float(result[3]) - 6.0) < 1e-10
    # y-1 = (1,0,1) = 1+0+1 = 2
    assert abs(float(result[4]) - 2.0) < 1e-10
    # z+1 = (1,1,2) = 1+1+4 = 6
    assert abs(float(result[5]) - 6.0) < 1e-10
    # z-1 = (1,1,0) = 1+1+0 = 2
    assert abs(float(result[6]) - 2.0) < 1e-10


def test_cell_accessor_traced_no_recompile():
    """Calling with different Nx, Ny, Nz values does not recompile."""
    from neon.blockamr.cell_accessor import CellAccessor

    ng = 1

    @jax.jit
    def read_center(buf, Nx, Ny, Nz):
        phi = CellAccessor(buf, 0, 0, Nx, Ny, Nz, ng)
        return phi.center

    # Shared buffer large enough for both box shapes
    buf = jnp.ones(400)

    # Call 1: 6x6x4
    _ = read_center(buf, jnp.int32(6), jnp.int32(6), jnp.int32(4))
    jax.block_until_ready(_)

    # Call 2: 10x10x4 — different dims, same buf shape → should NOT recompile
    compiles = [0]
    def count(ev, dur, **kw):
        if ev == "/jax/core/compile/backend_compile_duration":
            compiles[0] += 1
    jax.monitoring.register_event_duration_secs_listener(count)

    _ = read_center(buf, jnp.int32(10), jnp.int32(10), jnp.int32(4))
    jax.block_until_ready(_)
    jax.monitoring.unregister_event_duration_listener(count)

    assert compiles[0] == 0, (
        f"Expected 0 recompilations with traced Nx/Ny/Nz, got {compiles[0]}"
    )


def test_cell_accessor_traced_vmap():
    """CellAccessor works inside vmap with traced Nx, Ny, Nz."""
    from neon.blockamr.cell_accessor import CellAccessor

    Nx, Ny, Nz, ng = 6, 6, 4, 1
    n_cells = (Nx - 2*ng) * (Ny - 2*ng) * (Nz - 2*ng)
    box = _make_quad_box(Nx, Ny, Nz)

    @jax.jit
    def laplacian_vmap(buf, Nx, Ny, Nz):
        def one_cell(cell_idx):
            phi = CellAccessor(buf, 0, cell_idx, Nx, Ny, Nz, ng)
            return sum(
                phi.S(1, ax) - 2 * phi.center + phi.S(-1, ax)
                for ax in range(3)
            )
        return jax.vmap(one_cell)(jnp.arange(n_cells))

    result = laplacian_vmap(box, jnp.int32(Nx), jnp.int32(Ny), jnp.int32(Nz))
    # Laplacian of x^2+y^2+z^2 with dx=1 is 2+2+2=6 everywhere
    assert jnp.allclose(result, 6.0, atol=1e-10)
