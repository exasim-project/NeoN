# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Triton kernel library for structured grid stencils.

Provides:
  - phi(): Array4-like accessor for (i, j, k) indexing
  - triton_parallel_for(): launch wrapper for user @triton.jit kernels
  - _triton_wrapper(): 3D tiled kernel that calls a user stencil function

User writes:

    @triton.jit
    def laplacian(i, j, k, phi_ptr, sx, sy):
        c = phi(phi_ptr, i, j, k, sx, sy)
        return (phi(phi_ptr,i+1,j,k,sx,sy) - 2.*c + phi(phi_ptr,i-1,j,k,sx,sy)
              + phi(phi_ptr,i,j+1,k,sx,sy) - 2.*c + phi(phi_ptr,i,j-1,k,sx,sy)
              + phi(phi_ptr,i,j,k+1,sx,sy) - 2.*c + phi(phi_ptr,i,j,k-1,sx,sy))

    triton_parallel_for(laplacian, phi_tensor, out_tensor, Nx=66, Ny=66, Nz=66)
"""

import triton
import triton.language as tl


@triton.jit
def phi(ptr, i, j, k, sx: tl.constexpr, sy: tl.constexpr):
    """Array4-like accessor: phi(ptr, i, j, k, sx, sy) -> value.

    Computes flat offset from 3D index using C-order strides:
        flat = i * sx + j * sy + k
    where sx = Ny * Nz, sy = Nz.
    """
    return tl.load(ptr + i * sx + j * sy + k)


@triton.jit
def _triton_lap_wrapper(
    phi_ptr, out_ptr,
    sx: tl.constexpr, sy: tl.constexpr,
    osx: tl.constexpr, osy: tl.constexpr,
    ng: tl.constexpr, gx: tl.constexpr,
    box_vol: tl.constexpr, out_vol: tl.constexpr,
    TX: tl.constexpr, TY: tl.constexpr, TZ: tl.constexpr,
):
    """3D tiled kernel wrapper with fused box dimension in grid.x.

    grid = (n_boxes * gx, gy, gz)
    program_id(0) encodes both box_id and tile_x.
    """
    flat_id = tl.program_id(0)
    box_id = flat_id // gx
    tile_x = flat_id % gx
    tile_y = tl.program_id(1)
    tile_z = tl.program_id(2)

    ox = tile_x * TX
    oy = tile_y * TY
    oz = tile_z * TZ

    lx = tl.arange(0, TX)[:, None, None]
    ly = tl.arange(0, TY)[None, :, None]
    lz = tl.arange(0, TZ)[None, None, :]

    i = ng + ox + lx
    j = ng + oy + ly
    k = ng + oz + lz

    # Gather from ghosted input (box_id * box_vol offset)
    phi_base = phi_ptr + box_id * box_vol
    gi = i * sx + j * sy + k
    c = tl.load(phi_base + gi)
    xp = tl.load(phi_base + gi + sx)
    xm = tl.load(phi_base + gi - sx)
    yp = tl.load(phi_base + gi + sy)
    ym = tl.load(phi_base + gi - sy)
    zp = tl.load(phi_base + gi + 1)
    zm = tl.load(phi_base + gi - 1)

    val = ((xp - 2.0 * c + xm)
         + (yp - 2.0 * c + ym)
         + (zp - 2.0 * c + zm))

    # Store to output (box_id * out_vol offset)
    oi = box_id * out_vol + (ox + lx) * osx + (oy + ly) * osy + (oz + lz)
    tl.store(out_ptr + oi, val)


def triton_parallel_for_lap(phi_tensor, out_tensor, *,
                            Nx, Ny, Nz, ng=1, tile=8, num_warps=1,
                            n_boxes=1):
    """Launch the fused laplacian kernel over n_boxes with 3D tiling.

    Parameters
    ----------
    phi_tensor : torch.Tensor
        Flat contiguous buffer, shape (n_boxes * Nx * Ny * Nz,).
    out_tensor : torch.Tensor
        Flat output buffer, shape (n_boxes * bx * by * bz,).
    Nx, Ny, Nz : int
        Ghosted box dimensions.
    ng : int
        Ghost layers.
    tile : int
        Tile size per dimension.
    num_warps : int
        Triton warps per program.
    n_boxes : int
        Number of boxes (fused into grid.x).
    """
    bx = Nx - 2 * ng
    by = Ny - 2 * ng
    bz = Nz - 2 * ng
    sx = Ny * Nz
    sy = Nz
    osx = by * bz
    osy = bz
    box_vol = Nx * Ny * Nz
    out_vol = bx * by * bz
    gx = bx // tile
    gy = by // tile
    gz = bz // tile

    _triton_lap_wrapper[(n_boxes * gx, gy, gz)](
        phi_tensor, out_tensor,
        sx, sy, osx, osy, ng, gx, box_vol, out_vol,
        TX=tile, TY=tile, TZ=tile, num_warps=num_warps,
    )
