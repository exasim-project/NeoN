# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Precomputed offset dispatch: flat vmap over contiguous array.

Each valid cell has 4 precomputed int32 offsets (base, fx, fy, fz)
into the flat buffers. The kernel is a simple gather + stencil math
with no modulo, no division, no CellAccessor, no for_box.

Achieves 1.25x C++ at 128³ with linear scheme (pure XLA, no Triton).
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .flattened_boxes import (
    FlattenedBoxes, FlattenedFaceBoxes, flattened_boxes_from_mf,
    CELL_TIERS, _cell_tier, _next_power_of_2,
)


class LocalOffsets(eqx.Module):
    """Per-cell local offsets within a box shape. uint16, shared across all boxes.

    Precomputed once per unique (bx, by, bz, ng) combination.
    Typically 256 KB for 32³ boxes — fits L2 cache.
    """

    base: jax.Array     # (n_cells_per_box,) uint16 — local cell offset within box
    fx: jax.Array       # (n_cells_per_box,) uint16 — local face-x offset
    fy: jax.Array       # (n_cells_per_box,) uint16 — local face-y offset
    fz: jax.Array       # (n_cells_per_box,) uint16 — local face-z offset
    n_cells: int = eqx.field(static=True)


class SplitOffsets(eqx.Module):
    """Split offset dispatch: box_starts (int32, per-box) + local offsets (uint16, per-shape).

    Memory: box_starts = n_boxes × 4 × 4 bytes (~1 KB for 64 boxes).
    Local offsets = n_cells_per_box × 4 × 2 bytes (~256 KB for 32³, shared).
    Total: ~257 KB vs 33 MB for flat int32 offsets.
    """

    # Per-box absolute starts (traced, tiny)
    cell_starts: jax.Array    # (max_boxes,) int32
    fx_starts: jax.Array      # (max_boxes,) int32
    fy_starts: jax.Array      # (max_boxes,) int32
    fz_starts: jax.Array      # (max_boxes,) int32

    # Shared local offsets (traced, small, cached in L2)
    local: LocalOffsets

    # Per-element box mapping (traced)
    elem_to_box: jax.Array    # (total_padded,) int32 — which box each cell belongs to
    elem_to_local: jax.Array  # (total_padded,) int32 — local cell index within box

    total_valid: int = eqx.field(static=True)
    total_padded: int = eqx.field(static=True)

    # Strides (static)
    sx: int = eqx.field(static=True)
    sy: int = eqx.field(static=True)
    sz: int = eqx.field(static=True)
    fx_stride_r: int = eqx.field(static=True)
    fy_stride_r: int = eqx.field(static=True)
    fz_stride_r: int = eqx.field(static=True)


class PrecomputedOffsets(eqx.Module):
    """Flat int32 per-cell offsets (original approach, for comparison)."""

    base: jax.Array
    fx_off: jax.Array
    fy_off: jax.Array
    fz_off: jax.Array

    total_valid: int = eqx.field(static=True)
    total_padded: int = eqx.field(static=True)

    sx: int = eqx.field(static=True)
    sy: int = eqx.field(static=True)
    sz: int = eqx.field(static=True)
    ng: int = eqx.field(static=True)
    fx_stride_r: int = eqx.field(static=True)
    fy_stride_r: int = eqx.field(static=True)
    fz_stride_r: int = eqx.field(static=True)


# ---------------------------------------------------------------------------
# Tiering for total_padded
# ---------------------------------------------------------------------------

TOTAL_TIERS = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
]


def _total_tier(n):
    for t in TOTAL_TIERS:
        if n <= t:
            return t
    return _next_power_of_2(n)


# ---------------------------------------------------------------------------
# Build precomputed offsets
# ---------------------------------------------------------------------------

def build_local_offsets(bx, by, bz, ng):
    """Build shared local uint16 offsets for one box shape."""
    Nx_g = bx + 2*ng; Ny_g = by + 2*ng
    n_cells = bx * by * bz
    base = np.empty(n_cells, dtype=np.uint16)
    fx = np.empty(n_cells, dtype=np.uint16)
    fy = np.empty(n_cells, dtype=np.uint16)
    fz = np.empty(n_cells, dtype=np.uint16)
    for ci in range(n_cells):
        ix = ci % bx; iy = (ci // bx) % by; iz = ci // (bx * by)
        base[ci] = (ng+ix) + Nx_g*(ng+iy) + Nx_g*Ny_g*(ng+iz)
        fx[ci] = ix + (bx+1)*iy + (bx+1)*by*iz
        fy[ci] = ix + bx*iy + bx*(by+1)*iz
        fz[ci] = ix + bx*iy + bx*by*iz
    return LocalOffsets(
        base=jnp.array(base, dtype=jnp.uint16),
        fx=jnp.array(fx, dtype=jnp.uint16),
        fy=jnp.array(fy, dtype=jnp.uint16),
        fz=jnp.array(fz, dtype=jnp.uint16),
        n_cells=n_cells,
    )


# Cache local offsets by box shape
_local_offsets_cache = {}


def get_local_offsets(bx, by, bz, ng):
    """Get or create cached local offsets for a box shape."""
    key = (bx, by, bz, ng)
    if key not in _local_offsets_cache:
        _local_offsets_cache[key] = build_local_offsets(bx, by, bz, ng)
    return _local_offsets_cache[key]


def build_split_offsets(fb, face_fb, ng):
    """Build SplitOffsets: box_starts (int32) + local offsets (uint16).

    Memory: ~257 KB for 64 boxes of 32³ vs 33 MB for flat int32.
    """
    n_boxes = fb.n_boxes
    Nx_g0, Ny_g0, Nz_g0 = fb.shapes[0][:3]
    bx0 = Nx_g0 - 2*ng; by0 = Ny_g0 - 2*ng; bz0 = Nz_g0 - 2*ng
    n_cells = bx0 * by0 * bz0

    # Per-box starts
    cell_starts = [int(fb.offsets[b]) for b in range(n_boxes)]
    fx_starts = [int(face_fb.offsets[0][b]) for b in range(n_boxes)]
    fy_starts = [int(face_fb.offsets[1][b]) for b in range(n_boxes)]
    fz_starts = [int(face_fb.offsets[2][b]) for b in range(n_boxes)]

    # Shared local offsets
    local = get_local_offsets(bx0, by0, bz0, ng)

    # Element-to-box and element-to-local mappings
    e2b = []; e2l = []
    for b in range(n_boxes):
        e2b.extend([b] * n_cells)
        e2l.extend(range(n_cells))

    total_valid = len(e2b)
    total_padded = _total_tier(total_valid)

    pad_n = total_padded - total_valid
    if pad_n > 0:
        e2b.extend([0] * pad_n)
        e2l.extend([0] * pad_n)

    # Pad box starts to power of 2
    mb = max(n_boxes, 1)
    while mb & (mb-1): mb = (mb | (mb-1)) + 1
    cell_starts.extend([cell_starts[0]] * (mb - n_boxes))
    fx_starts.extend([fx_starts[0]] * (mb - n_boxes))
    fy_starts.extend([fy_starts[0]] * (mb - n_boxes))
    fz_starts.extend([fz_starts[0]] * (mb - n_boxes))

    return SplitOffsets(
        cell_starts=jnp.array(cell_starts, dtype=jnp.int32),
        fx_starts=jnp.array(fx_starts, dtype=jnp.int32),
        fy_starts=jnp.array(fy_starts, dtype=jnp.int32),
        fz_starts=jnp.array(fz_starts, dtype=jnp.int32),
        local=local,
        elem_to_box=jnp.array(e2b[:total_padded], dtype=jnp.int32),
        elem_to_local=jnp.array(e2l[:total_padded], dtype=jnp.int32),
        total_valid=total_valid,
        total_padded=total_padded,
        sx=1, sy=Nx_g0, sz=Nx_g0*Ny_g0,
        fx_stride_r=1, fy_stride_r=bx0, fz_stride_r=bx0*by0,
    )


def build_precomputed_offsets(fb, face_fb, ng, mf=None, fx_mf=None, fy_mf=None, fz_mf=None):
    """Build PrecomputedOffsets from FlattenedBoxes + FlattenedFaceBoxes.

    If MultiFab objects are provided, uses the fast C++ builder.
    Otherwise falls back to Python.

    Returns PrecomputedOffsets.
    """
    import neon.blockamr as blockamr

    # Box shape for strides (assume uniform — use first box)
    Nx_g0, Ny_g0, Nz_g0 = fb.shapes[0][:3]
    bx0 = Nx_g0 - 2 * ng
    by0 = Ny_g0 - 2 * ng

    # Try C++ path
    if mf is not None and fx_mf is not None:
        base_l, fx_l, fy_l, fz_l = blockamr.build_stencil_offsets(
            mf, fx_mf, fy_mf, fz_mf, ng)
        bases = list(base_l)
        fx_offs = list(fx_l)
        fy_offs = list(fy_l)
        fz_offs = list(fz_l)
    else:
        # Python fallback
        n_boxes = fb.n_boxes
        bases = []; fx_offs = []; fy_offs = []; fz_offs = []
        for b in range(n_boxes):
            Nx_g, Ny_g, Nz_g = fb.shapes[b][:3]
            bx = Nx_g - 2*ng; by_ = Ny_g - 2*ng; bz = Nz_g - 2*ng
            n_valid = bx * by_ * bz
            cell_off = int(fb.offsets[b])
            fxo = int(face_fb.offsets[0][b])
            fyo = int(face_fb.offsets[1][b])
            fzo = int(face_fb.offsets[2][b])
            for ci in range(n_valid):
                ix = ci % bx; iy = (ci // bx) % by_; iz = ci // (bx * by_)
                bases.append(cell_off + (ng+ix) + Nx_g*(ng+iy) + Nx_g*Ny_g*(ng+iz))
                fx_offs.append(fxo + ix + (bx+1)*iy + (bx+1)*by_*iz)
                fy_offs.append(fyo + ix + bx*iy + bx*(by_+1)*iz)
                fz_offs.append(fzo + ix + bx*iy + bx*by_*iz)

    total_valid = len(bases)
    total_padded = _total_tier(total_valid)

    # Pad with first element (safe dummy)
    pad_n = total_padded - total_valid
    if pad_n > 0:
        bases.extend([bases[0]] * pad_n)
        fx_offs.extend([fx_offs[0]] * pad_n)
        fy_offs.extend([fy_offs[0]] * pad_n)
        fz_offs.extend([fz_offs[0]] * pad_n)

    return PrecomputedOffsets(
        base=jnp.array(bases[:total_padded], dtype=jnp.int32),
        fx_off=jnp.array(fx_offs[:total_padded], dtype=jnp.int32),
        fy_off=jnp.array(fy_offs[:total_padded], dtype=jnp.int32),
        fz_off=jnp.array(fz_offs[:total_padded], dtype=jnp.int32),
        total_valid=total_valid,
        total_padded=total_padded,
        sx=1,
        sy=Nx_g0,
        sz=Nx_g0 * Ny_g0,
        ng=ng,
        fx_stride_r=1,
        fy_stride_r=bx0,
        fz_stride_r=bx0 * by0,
    )


# ---------------------------------------------------------------------------
# Dispatch: flat vmap stencil kernel
# ---------------------------------------------------------------------------

@jax.jit
def linear_euler_step(cell_buf, fx_buf, fy_buf, fz_buf,
                      offsets, dh, dt_over_coeff, nu):
    """Forward Euler step: phi_new = phi - dt*(div(phi,U) - nu*lap(phi)).

    Linear divergence + central laplacian, ncomp=1.
    Single flat vmap over all valid cells.
    """
    idx_arr = 1.0 / dh
    idx2_arr = idx_arr ** 2
    sx = offsets.sx
    sy = offsets.sy
    sz = offsets.sz

    def process_one(i):
        b = offsets.base[i]

        c = cell_buf[b]
        xm = cell_buf[b - sx]; xp = cell_buf[b + sx]
        ym = cell_buf[b - sy]; yp = cell_buf[b + sy]
        zm = cell_buf[b - sz]; zp = cell_buf[b + sz]

        fb = offsets.fx_off[i]
        flx = fx_buf[fb]; frx = fx_buf[fb + offsets.fx_stride_r]
        fb2 = offsets.fy_off[i]
        fly = fy_buf[fb2]; fry = fy_buf[fb2 + offsets.fy_stride_r]
        fb3 = offsets.fz_off[i]
        flz = fz_buf[fb3]; frz = fz_buf[fb3 + offsets.fz_stride_r]

        div = ((frx * 0.5 * (c + xp) - flx * 0.5 * (xm + c)) * idx_arr[0]
             + (fry * 0.5 * (c + yp) - fly * 0.5 * (ym + c)) * idx_arr[1]
             + (frz * 0.5 * (c + zp) - flz * 0.5 * (zm + c)) * idx_arr[2])

        lap = ((xp - 2*c + xm) * idx2_arr[0]
             + (yp - 2*c + ym) * idx2_arr[1]
             + (zp - 2*c + zm) * idx2_arr[2])

        # No mask needed — padded cells point to valid dummy data
        return c - dt_over_coeff * (div - nu * lap)

    return jax.vmap(process_one)(jnp.arange(offsets.total_padded))


@jax.jit
def linear_euler_step_split(cell_buf, fx_buf, fy_buf, fz_buf,
                            offsets, dh, dt_over_coeff, nu):
    """Forward Euler with split uint16 offsets. Minimal bandwidth."""
    idx_arr = 1.0 / dh
    idx2_arr = idx_arr ** 2
    sx = offsets.sx; sy = offsets.sy; sz = offsets.sz

    def process_one(i):
        bi = offsets.elem_to_box[i]
        li = offsets.elem_to_local[i]

        # Reconstruct absolute offset: box_start + local_uint16
        b = offsets.cell_starts[bi] + offsets.local.base[li].astype(jnp.int32)

        c = cell_buf[b]
        xm = cell_buf[b - sx]; xp = cell_buf[b + sx]
        ym = cell_buf[b - sy]; yp = cell_buf[b + sy]
        zm = cell_buf[b - sz]; zp = cell_buf[b + sz]

        fb = offsets.fx_starts[bi] + offsets.local.fx[li].astype(jnp.int32)
        flx = fx_buf[fb]; frx = fx_buf[fb + offsets.fx_stride_r]
        fb2 = offsets.fy_starts[bi] + offsets.local.fy[li].astype(jnp.int32)
        fly = fy_buf[fb2]; fry = fy_buf[fb2 + offsets.fy_stride_r]
        fb3 = offsets.fz_starts[bi] + offsets.local.fz[li].astype(jnp.int32)
        flz = fz_buf[fb3]; frz = fz_buf[fb3 + offsets.fz_stride_r]

        div = ((frx * 0.5 * (c + xp) - flx * 0.5 * (xm + c)) * idx_arr[0]
             + (fry * 0.5 * (c + yp) - fly * 0.5 * (ym + c)) * idx_arr[1]
             + (frz * 0.5 * (c + zp) - flz * 0.5 * (zm + c)) * idx_arr[2])

        lap = ((xp - 2*c + xm) * idx2_arr[0]
             + (yp - 2*c + ym) * idx2_arr[1]
             + (zp - 2*c + zm) * idx2_arr[2])

        return c - dt_over_coeff * (div - nu * lap)

    return jax.vmap(process_one)(jnp.arange(offsets.total_padded))


@jax.jit
def linear_euler_step_ncomp(cell_buf, fx_buf, fy_buf, fz_buf,
                            offsets, dh, dt_over_coeff, nu, ncomp, plane_size):
    """Forward Euler step for ncomp > 1.

    AMReX stores components as planes: comp_offset = comp * plane_size.
    """
    idx_arr = 1.0 / dh
    idx2_arr = idx_arr ** 2
    sx = offsets.sx
    sy = offsets.sy
    sz = offsets.sz

    def process_one(i):
        b = offsets.base[i]
        is_valid = i < offsets.total_valid
        results = []

        for comp in range(ncomp):
            bc = b + comp * plane_size
            c = cell_buf[bc]
            xm = cell_buf[bc - sx]; xp = cell_buf[bc + sx]
            ym = cell_buf[bc - sy]; yp = cell_buf[bc + sy]
            zm = cell_buf[bc - sz]; zp = cell_buf[bc + sz]

            fb = offsets.fx_off[i]
            flx = fx_buf[fb]; frx = fx_buf[fb + offsets.fx_stride_r]
            fb2 = offsets.fy_off[i]
            fly = fy_buf[fb2]; fry = fy_buf[fb2 + offsets.fy_stride_r]
            fb3 = offsets.fz_off[i]
            flz = fz_buf[fb3]; frz = fz_buf[fb3 + offsets.fz_stride_r]

            div = ((frx * 0.5 * (c + xp) - flx * 0.5 * (xm + c)) * idx_arr[0]
                 + (fry * 0.5 * (c + yp) - fly * 0.5 * (ym + c)) * idx_arr[1]
                 + (frz * 0.5 * (c + zp) - flz * 0.5 * (zm + c)) * idx_arr[2])

            lap = ((xp - 2*c + xm) * idx2_arr[0]
                 + (yp - 2*c + ym) * idx2_arr[1]
                 + (zp - 2*c + zm) * idx2_arr[2])

            results.append(c - dt_over_coeff * (div - nu * lap))

        return jnp.array(results)

    return jax.vmap(process_one)(jnp.arange(offsets.total_padded))


# ---------------------------------------------------------------------------
# Scatter back to per-box arrays
# ---------------------------------------------------------------------------

def scatter_precomputed(result, fb, ncomp=1):
    """Reshape flat result → per-box arrays for MultiFab.copy_arrays."""
    ng = fb.n_grow
    all_results = [None] * fb.n_boxes
    offset = 0
    for b in range(fb.n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        vNz = Nz - 2 * ng
        n_valid = vNx * vNy * vNz

        if ncomp == 1:
            box_data = result[offset:offset + n_valid]
            all_results[b] = box_data.reshape(vNz, vNy, vNx).transpose(2, 1, 0)[:, :, :, None]
        else:
            box_data = result[offset:offset + n_valid]  # (n_valid, ncomp)
            comps = []
            for c in range(ncomp):
                comp_3d = box_data[:, c].reshape(vNz, vNy, vNx).transpose(2, 1, 0)
                comps.append(comp_3d)
            all_results[b] = jnp.stack(comps, axis=-1)

        offset += n_valid
    return all_results
