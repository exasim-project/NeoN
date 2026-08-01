# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for FlattenedBoxes constructed from real MultiFab data."""

import jax
import jax.numpy as jnp

import blockamr
from blockamr.mesh import Mesh
from blockamr.field import CellField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.flattened_boxes import (
    FlattenedBoxes,
    flattened_boxes_from_mf,
    build_buckets,
)
from blockamr.cell_kernels import CellLaplacianKernel
from blockamr.bucket_dispatch import process_bucket


def _make_periodic_mesh(N, Nz=None, max_size=None):
    """Fully periodic mesh on [0,1]^2 x [0, Nz/N]."""
    Nz = Nz or N
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def test_flattened_boxes_from_mf():
    """FlattenedBoxes.from_mf constructs from a real MultiFab."""
    mesh, geom = _make_periodic_mesh(8, Nz=4, max_size=4)
    field = CellField(mesh, ncomp=1, ngrow=1, name="test",
                       fill_patch=FillPatchCellConservative())

    # Fill with known values
    mf = field.mf[0]
    mf.set_val(42.0)

    fb = flattened_boxes_from_mf(mf)

    # Check types
    assert isinstance(fb, FlattenedBoxes)
    assert fb.contiguous_array.ndim == 1
    assert fb.offsets.ndim == 1
    assert isinstance(fb.shapes, tuple)
    assert fb.n_grow == 1

    # With 8x8x4 and max_size=4: should have multiple boxes
    n_boxes = len(fb.offsets)
    assert n_boxes > 1

    # Each box should have grown dims = valid + 2*ngrow
    meta = mf.fab_metadata()
    for b in range(n_boxes):
        assert fb.shapes[b][0] == meta[b][1]  # Nx
        assert fb.shapes[b][1] == meta[b][2]  # Ny
        assert fb.shapes[b][2] == meta[b][3]  # Nz
        assert fb.shapes[b][3] == meta[b][4]  # nc

    # Data should match: the value 42.0 should be in the buffer
    assert float(jnp.max(fb.contiguous_array)) == 42.0


def test_bucket_dispatch_with_real_multifab_simple():
    """Verify CellAccessor reads correct values from real MultiFab."""
    N = 8
    Nz = 4
    max_size = 8  # single box — simplest case
    ng = 1
    mesh, geom = _make_periodic_mesh(N, Nz=Nz, max_size=max_size)
    field = CellField(mesh, ncomp=1, ngrow=ng, name="u",
                       fill_patch=FillPatchCellConservative())
    dx = geom.cell_size()

    # Fill valid region with u(i,j,k) = i^2 + j^2 + k^2 (cell indices)
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        vals = jnp.zeros((nx, ny, nz, 1))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    gi = lo[0] + i
                    gj = lo[1] + j
                    gk = lo[2] + k
                    vals = vals.at[i, j, k, 0].set(float(gi**2 + gj**2 + gk**2))
        mf.copy_from(mfi, vals)

    field.fill_patch(0, 0.0)

    fb = flattened_boxes_from_mf(mf)
    meta = mf.fab_metadata()

    # Single box: check that CellAccessor reads the correct center value
    from blockamr.cell_accessor import CellAccessor
    Nx_g, Ny_g, Nz_g = meta[0][1], meta[0][2], meta[0][3]

    # cell_idx=0 → (ng, ng, ng) in grown box = valid cell (0,0,0) in global coords
    phi = CellAccessor(fb.contiguous_array, int(fb.offsets[0]),
                        0, Nx_g, Ny_g, Nz_g, ng)
    # Global cell (0,0,0): value = 0^2 + 0^2 + 0^2 = 0
    assert abs(float(phi.center)) < 1e-10, f"center = {float(phi.center)}"

    # cell_idx=1 → (ng+1, ng, ng) = valid cell (1,0,0): value = 1
    phi1 = CellAccessor(fb.contiguous_array, int(fb.offsets[0]),
                         1, Nx_g, Ny_g, Nz_g, ng)
    assert abs(float(phi1.center) - 1.0) < 1e-10, f"center = {float(phi1.center)}"


def test_bucket_dispatch_laplacian_periodic():
    """Laplacian of u = sin(2πx) via bucket dispatch on real MultiFab.

    sin(2πx) is periodic → ghost cells are correct.
    Discrete lap in x: (sin(2π(x+dx)) - 2 sin(2πx) + sin(2π(x-dx))) / dx^2.
    For fine enough grid, this → -(2π)^2 sin(2πx).
    """
    import math

    N = 16
    Nz = 4
    ng = 1
    mesh, geom = _make_periodic_mesh(N, Nz=Nz, max_size=N)
    field = CellField(mesh, ncomp=1, ngrow=ng, name="u",
                       fill_patch=FillPatchCellConservative())
    dx_phys = geom.cell_size()

    # Fill with u = sin(2πx) (only varies in x, constant in y,z)
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        vals = jnp.zeros((nx, ny, nz, 1))
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx_phys[0]
            for j in range(ny):
                for k in range(nz):
                    vals = vals.at[i, j, k, 0].set(
                        math.sin(2 * math.pi * x)
                    )
        mf.copy_from(mfi, vals)

    field.fill_patch(0, 0.0)

    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in dx_phys)
    buckets = build_buckets(fb, dh)
    kernel = CellLaplacianKernel(dh=dh, coeff=1.0)

    for bucket in buckets:
        if bucket.n_valid == 0:
            continue

        result0 = process_bucket(bucket, 0.0, (kernel,))
        result1 = process_bucket(bucket, 1.0, (kernel,))
        # lap = result0 - result1
        lap = result0[:bucket.n_valid] - result1[:bucket.n_valid]

        # Analytical: lap(sin(2πx)) = -(2π)^2 sin(2πx)
        # But discrete laplacian differs. The exact discrete value is:
        # (sin(2π(x+dx)) - 2sin(2πx) + sin(2π(x-dx))) / dx^2
        # = 2*(cos(2πdx) - 1) / dx^2 * sin(2πx)
        # So lap / center = 2*(cos(2πdx) - 1) / dx^2 (a constant)
        discrete_factor = 2.0 * (
            math.cos(2 * math.pi * dx_phys[0]) - 1
        ) / dx_phys[0]**2

        # Where center != 0, check ratio lap/center ≈ discrete_factor
        centers = result0[:bucket.n_valid]
        mask = jnp.abs(centers) > 0.1  # avoid division by near-zero
        if jnp.any(mask):
            ratios = lap[mask] / centers[mask]
            assert jnp.allclose(ratios, discrete_factor, atol=1e-6), (
                f"max ratio error = "
                f"{float(jnp.max(jnp.abs(ratios - discrete_factor)))}"
            )
