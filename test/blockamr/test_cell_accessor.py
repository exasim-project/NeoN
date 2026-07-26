# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for cell accessor classes (StencilAxis, CellAccessor, FaceAccessor)."""

import jax
import jax.numpy as jnp

from blockamr.cell_accessor import StencilAxis, CellAccessor, FaceAccessor
from blockamr.cell_kernels import (
    CellLaplacianKernel,
    CellUpwindDivKernel,
    CellVanLeerDivKernel,
)


def _make_quad_box(Nx, Ny, Nz, dx=1.0):
    """Fill a box with u = (i*dx)^2 + (j*dx)^2 + (k*dx)^2 in Fortran order."""
    buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                buf = buf.at[i + Nx * j + Nx * Ny * k].set(
                    (i * dx) ** 2 + (j * dx) ** 2 + (k * dx) ** 2
                )
    return buf


# --- StencilAxis ---

def test_stencil_axis_reads_neighbors():
    buf = jnp.arange(20, dtype=jnp.float64)
    axis = StencilAxis(buf, base=5, stride=2)
    assert float(axis[0]) == 5.0
    assert float(axis[1]) == 7.0
    assert float(axis[-1]) == 3.0
    assert float(axis[2]) == 9.0


def test_stencil_axis_works_in_vmap():
    buf = jnp.arange(20, dtype=jnp.float64)

    @jax.jit
    def f(buf, offsets):
        def read(off):
            ax = StencilAxis(buf, off, stride=1)
            return ax[0] + ax[1]
        return jax.vmap(read)(offsets)

    offsets = jnp.array([2, 5, 10], dtype=jnp.int32)
    expected = jnp.array([2 + 3, 5 + 6, 10 + 11], dtype=jnp.float64)
    assert jnp.allclose(f(buf, offsets), expected)


# --- CellAccessor ---

def test_cell_accessor_center_and_neighbors():
    Nx, Ny, Nz, ng = 6, 6, 4, 1
    buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                buf = buf.at[i + Nx * j + Nx * Ny * k].set(
                    float(i + 10 * j + 100 * k)
                )
    phi = CellAccessor(buf, 0, 0, Nx, Ny, Nz, ng)
    assert float(phi.center) == 111.0
    assert float(phi.x[1]) == 112.0
    assert float(phi.S(1, 0)) == 112.0
    assert float(phi.x[-1]) == 110.0
    assert float(phi.y[1]) == 121.0
    assert float(phi.z[1]) == 211.0


def test_cell_accessor_rectangular_box():
    Nx, Ny, Nz, ng = 10, 6, 4, 1
    buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                buf = buf.at[i + Nx * j + Nx * Ny * k].set(
                    float(i + 10 * j + 100 * k)
                )
    phi = CellAccessor(buf, 0, 0, Nx, Ny, Nz, ng)
    assert float(phi.center) == 111.0
    vNx = Nx - 2 * ng
    phi2 = CellAccessor(buf, 0, vNx, Nx, Ny, Nz, ng)
    assert float(phi2.center) == 121.0


def test_cell_accessor_laplacian_in_vmap():
    Nx, Ny, Nz, ng = 8, 8, 6, 1
    buf = _make_quad_box(Nx, Ny, Nz)
    n_cells = (Nx - 2 * ng) * (Ny - 2 * ng) * (Nz - 2 * ng)
    dx = jnp.array([1.0, 1.0, 1.0])

    @jax.jit
    def compute(buf, dx):
        def lap(idx):
            phi = CellAccessor(buf, 0, idx, Nx, Ny, Nz, ng)
            return sum(
                (phi.S(1, ax) - 2 * phi.center + phi.S(-1, ax)) / dx[ax]**2
                for ax in range(3)
            )
        return jax.vmap(lap)(jnp.arange(n_cells))

    assert jnp.allclose(compute(buf, dx), 6.0)


def test_cell_accessor_nested_vmap_multiple_boxes():
    Nx, Ny, Nz, ng = 6, 6, 4, 1
    n_cells = (Nx - 2 * ng) * (Ny - 2 * ng) * (Nz - 2 * ng)
    dx = jnp.array([1.0, 1.0, 1.0])
    buf = jnp.concatenate([_make_quad_box(Nx, Ny, Nz)] * 2)
    offsets = jnp.array([0, Nx * Ny * Nz], dtype=jnp.int32)

    @jax.jit
    def compute(buf, offsets, dx):
        def box(off):
            def lap(idx):
                phi = CellAccessor(buf, off, idx, Nx, Ny, Nz, ng)
                return sum(
                    (phi.S(1, ax) - 2 * phi.center + phi.S(-1, ax)) / dx[ax]**2
                    for ax in range(3)
                )
            return jax.vmap(lap)(jnp.arange(n_cells))
        return jax.vmap(box)(offsets)

    result = compute(buf, offsets, dx)
    assert result.shape == (2, n_cells)
    assert jnp.allclose(result, 6.0)


# --- FaceAccessor ---

def test_face_accessor_xyz_indexing():
    Nx, Ny, Nz, ng = 6, 4, 4, 1
    fx_buf = jnp.arange((Nx + 1) * Ny * Nz, dtype=jnp.float64)
    fy_buf = jnp.arange(Nx * (Ny + 1) * Nz, dtype=jnp.float64) + 1000

    ff = FaceAccessor(
        face_bufs=(fx_buf, fy_buf, jnp.zeros(1)),
        face_offsets=(0, 0, 0), cell_idx=0, Nx=Nx, Ny=Ny, Nz=Nz, ng=ng,
    )
    Nx_f = Nx + 1
    assert float(ff.x[0]) == float(fx_buf[1 + Nx_f * 1 + Nx_f * Ny * 1])
    assert float(ff.x[1]) == float(fx_buf[2 + Nx_f * 1 + Nx_f * Ny * 1])

    Ny_f = Ny + 1
    assert float(ff.y[0]) == float(fy_buf[1 + Nx * 1 + Nx * Ny_f * 1])
    assert float(ff.y[1]) == float(fy_buf[1 + Nx * 2 + Nx * Ny_f * 1])


def test_face_accessor_non_uniform_multibox():
    """Verify per-direction offsets are correct when Nx != Ny != Nz.

    This is the bug scenario: x/y/z face MultiFabs have different grown
    box sizes, so their offsets into the contiguous buffer differ.
    E.g. with Nx=10, Ny=8, Nz=6 (grown), ng=2:
      x-face box size = 11 * 8 * 6 = 528
      y-face box size = 10 * 9 * 6 = 540
      z-face box size = 10 * 8 * 7 = 560
    Using x-face offsets for all directions would read wrong data for box > 0.
    """
    Nx, Ny, Nz, ng = 10, 8, 6, 2

    fx_box_size = (Nx + 1) * Ny * Nz  # 528
    fy_box_size = Nx * (Ny + 1) * Nz  # 540
    fz_box_size = Nx * Ny * (Nz + 1)  # 560

    n_boxes = 2
    # Fill each box with distinct values: box b has base value b*1000
    fx_buf = jnp.concatenate([jnp.arange(fx_box_size, dtype=jnp.float64) + b * 1000
                              for b in range(n_boxes)])
    fy_buf = jnp.concatenate([jnp.arange(fy_box_size, dtype=jnp.float64) + b * 1000 + 10000
                              for b in range(n_boxes)])
    fz_buf = jnp.concatenate([jnp.arange(fz_box_size, dtype=jnp.float64) + b * 1000 + 20000
                              for b in range(n_boxes)])

    # Per-direction offsets for box 1
    fx_off1 = fx_box_size  # 528
    fy_off1 = fy_box_size  # 540
    fz_off1 = fz_box_size  # 560
    # These are all DIFFERENT — this is the key: a single offset doesn't work.

    # Access cell 0 in box 1
    ff = FaceAccessor(
        face_bufs=(fx_buf, fy_buf, fz_buf),
        face_offsets=(fx_off1, fy_off1, fz_off1),
        cell_idx=0, Nx=Nx, Ny=Ny, Nz=Nz, ng=ng,
    )

    # cell_idx=0 maps to (i=ng, j=ng, k=ng) = (2, 2, 2)
    i, j, k = ng, ng, ng

    # x-face: expected base = fx_off1 + i + (Nx+1)*j + (Nx+1)*Ny*k
    Nx_f = Nx + 1
    expected_fx = fx_buf[fx_off1 + i + Nx_f * j + Nx_f * Ny * k]
    assert float(ff.x[0]) == float(expected_fx)

    # y-face: expected base = fy_off1 + i + Nx*j + Nx*(Ny+1)*k
    Ny_f = Ny + 1
    expected_fy = fy_buf[fy_off1 + i + Nx * j + Nx * Ny_f * k]
    assert float(ff.y[0]) == float(expected_fy)

    # z-face: expected base = fz_off1 + i + Nx*j + Nx*Ny*k
    expected_fz = fz_buf[fz_off1 + i + Nx * j + Nx * Ny * k]
    assert float(ff.z[0]) == float(expected_fz)

    # Verify that using a WRONG single offset would fail for z-faces.
    # If we used fx_off1 (528) instead of fz_off1 (560) for z,
    # we'd read fz_buf[528 + ...] instead of fz_buf[560 + ...].
    wrong_fz = fz_buf[fx_off1 + i + Nx * j + Nx * Ny * k]
    assert float(expected_fz) != float(wrong_fz), \
        "Test is degenerate — x and z offsets are equal, can't detect the bug"


# --- CentralDiffLaplacianKernel ---

def test_laplacian_kernel_callable():
    Nx, Ny, Nz, ng = 8, 8, 6, 1
    dx = jnp.array([0.5, 0.5, 0.5])
    buf = _make_quad_box(Nx, Ny, Nz, dx=0.5)
    kernel = CellLaplacianKernel(dh=tuple(float(d) for d in dx), coeff=1.0)
    phi = CellAccessor(buf, 0, 0, Nx, Ny, Nz, ng)
    assert abs(float(kernel(phi)) - 6.0) < 1e-10


# --- UpwindDivKernel ---

def test_upwind_div_kernel_constant_field():
    """Upwind div of a constant field = 0."""
    Nx, Ny, Nz, ng = 8, 8, 6, 1
    dx = jnp.array([1.0, 1.0, 1.0])

    # Constant cell field u = 5
    cell_buf = jnp.full(Nx * Ny * Nz, 5.0)

    # Constant face flux f = 1 in all directions
    fx_buf = jnp.ones((Nx + 1) * Ny * Nz)
    fy_buf = jnp.ones(Nx * (Ny + 1) * Nz)
    fz_buf = jnp.ones(Nx * Ny * (Nz + 1))

    phi = CellAccessor(cell_buf, 0, 0, Nx, Ny, Nz, ng)
    kernel = CellUpwindDivKernel(
        face_bufs=(fx_buf, fy_buf, fz_buf),
        face_offsets=(jnp.zeros(1, dtype=jnp.int32),) * 3, _face_offset=(0, 0, 0),
        Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, dh=(1.0, 1.0, 1.0), coeff=1.0,
    )

    result = kernel(phi)
    assert abs(float(result)) < 1e-10, f"got {float(result)}"


def test_upwind_div_kernel_linear_field():
    """Upwind div of linear field u = i with positive flux = 1/dx in x."""
    Nx, Ny, Nz, ng = 8, 8, 6, 1
    dx = jnp.array([1.0, 1.0, 1.0])

    cell_buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                cell_buf = cell_buf.at[i + Nx * j + Nx * Ny * k].set(float(i))

    fx_buf = jnp.ones((Nx + 1) * Ny * Nz)
    fy_buf = jnp.ones(Nx * (Ny + 1) * Nz)
    fz_buf = jnp.ones(Nx * Ny * (Nz + 1))

    phi = CellAccessor(cell_buf, 0, 0, Nx, Ny, Nz, ng)
    kernel = CellUpwindDivKernel(
        face_bufs=(fx_buf, fy_buf, fz_buf),
        face_offsets=(jnp.zeros(1, dtype=jnp.int32),) * 3, _face_offset=(0, 0, 0),
        Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, dh=(1.0, 1.0, 1.0), coeff=1.0,
    )

    result = kernel(phi)
    assert abs(float(result) - 1.0) < 1e-10, f"got {float(result)}"


# --- VanLeerDivKernel ---

def test_vanleer_div_kernel_constant_field():
    """VanLeer div of a constant field = 0."""
    Nx, Ny, Nz, ng = 10, 10, 6, 2  # ng=2 for VanLeer stencil width
    dx = jnp.array([1.0, 1.0, 1.0])

    cell_buf = jnp.full(Nx * Ny * Nz, 5.0)
    fx_buf = jnp.ones((Nx + 1) * Ny * Nz)
    fy_buf = jnp.ones(Nx * (Ny + 1) * Nz)
    fz_buf = jnp.ones(Nx * Ny * (Nz + 1))

    phi = CellAccessor(cell_buf, 0, 0, Nx, Ny, Nz, ng)
    kernel = CellVanLeerDivKernel(
        face_bufs=(fx_buf, fy_buf, fz_buf),
        face_offsets=(jnp.zeros(1, dtype=jnp.int32),) * 3, _face_offset=(0, 0, 0),
        Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, dh=(1.0, 1.0, 1.0), coeff=1.0,
    )

    result = kernel(phi)
    assert abs(float(result)) < 1e-10, f"got {float(result)}"


def test_vanleer_div_kernel_linear_field():
    """VanLeer div of linear field should match upwind (no limiting needed)."""
    Nx, Ny, Nz, ng = 10, 10, 6, 2
    dx = jnp.array([1.0, 1.0, 1.0])

    cell_buf = jnp.zeros(Nx * Ny * Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                cell_buf = cell_buf.at[i + Nx * j + Nx * Ny * k].set(float(i))

    fx_buf = jnp.ones((Nx + 1) * Ny * Nz)
    fy_buf = jnp.ones(Nx * (Ny + 1) * Nz)
    fz_buf = jnp.ones(Nx * Ny * (Nz + 1))

    phi = CellAccessor(cell_buf, 0, 0, Nx, Ny, Nz, ng)
    kernel = CellVanLeerDivKernel(
        face_bufs=(fx_buf, fy_buf, fz_buf),
        face_offsets=(jnp.zeros(1, dtype=jnp.int32),) * 3, _face_offset=(0, 0, 0),
        Nx=Nx, Ny=Ny, Nz=Nz, ng=ng, dh=(1.0, 1.0, 1.0), coeff=1.0,
    )

    result = kernel(phi)
    assert abs(float(result) - 1.0) < 1e-10, f"got {float(result)}"
