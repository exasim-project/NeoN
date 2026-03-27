# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Flat contiguous buffer data structures for bucketed dispatch."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class FlattenedBoxes(eqx.Module):
    """Contiguous MultiFab data with per-box metadata.

    eqx.Module — shapes and n_grow are static (trigger recompile when changed),
    contiguous_array and offsets are traced leaves.
    """

    contiguous_array: Array  # (total_elems,) flat 1D buffer
    offsets: Array  # (n_boxes,) int — start offset of each box
    shapes: tuple = eqx.field(static=True)  # ((Nx,Ny,Nz,nc), ...) per box
    n_grow: int = eqx.field(static=True)  # ghost cell layers

    @property
    def n_boxes(self):
        return len(self.offsets)


def flattened_boxes_from_mf(mf):
    """Construct FlattenedBoxes from a MultiFab.

    Uses contiguous_array() (zero-copy) and fab_metadata() from the
    C++ bindings.
    """
    values = mf.contiguous_array()
    meta = mf.fab_metadata()
    offsets = jnp.array([m[0] for m in meta], dtype=jnp.int32)
    shapes = tuple((m[1], m[2], m[3], m[4]) for m in meta)
    return FlattenedBoxes(
        contiguous_array=values, offsets=offsets, shapes=shapes,
        n_grow=mf.n_grow(),
    )


class BucketContext(eqx.Module):
    """A group of boxes bucketed by cell-count tier for vectorised dispatch.

    Static fields are tier constants (fixed ceilings) and ng (uniform).
    Traced fields hold per-box geometry and data arrays — these change on
    regrid without triggering JAX recompilation.
    """

    box_offsets: Array  # traced: (max_boxes,) box starts in cell_buf
    cell_buf: Array  # traced: flat cell buffer (level-wide)
    Nx_arr: Array  # traced: (max_boxes,) per-box grown x-dim
    Ny_arr: Array  # traced: (max_boxes,) per-box grown y-dim
    Nz_arr: Array  # traced: (max_boxes,) per-box grown z-dim
    n_cells_arr: Array  # traced: (max_boxes,) per-box valid cell count
    dh_arr: Array  # traced: (max_boxes, 3) per-box cell spacing
    ng: int = eqx.field(static=True)  # ghost cells (uniform)
    n_cells_padded: int = eqx.field(static=True)  # tier ceiling for inner vmap
    max_boxes: int = eqx.field(static=True)  # fixed max box count for outer vmap
    n_valid: int = eqx.field(static=True)  # actual valid box count
    box_indices: tuple = eqx.field(static=True)  # mapping to MFIterator order
    lev: int = eqx.field(static=True, default=0)  # AMR level


class FlattenedFaceBoxes(eqx.Module):
    """Flat face-field data for all directions.

    bufs: tuple of 3 flat 1D arrays (fx, fy, fz) — traced.
    offsets: tuple of 3 offset arrays — traced.
    """

    bufs: tuple  # (fx_buf, fy_buf, fz_buf)
    offsets: tuple  # (fx_offsets, fy_offsets, fz_offsets)

    @staticmethod
    def from_face_field(face_field, lev):
        """Construct from a FaceField at the given level."""
        face_lev = face_field[lev]
        bufs = tuple(face_lev[d].mf.contiguous_array() for d in range(3))
        offsets = tuple(
            jnp.array(
                [m[0] for m in face_lev[d].mf.fab_metadata()], dtype=jnp.int32
            )
            for d in range(3)
        )
        return FlattenedFaceBoxes(bufs=bufs, offsets=offsets)


def pad_buffer(buf, box_size=None):
    """Pad a 1D buffer so its length is a power-of-2 multiple of box_size.

    If box_size is given: pads to next_pow2(n_boxes) * box_size.
    If box_size is None: pads to next_pow2(len(buf)).

    This ensures that buffers with similar box counts share the same
    padded shape, limiting JAX recompilation to ~log2(N) unique sizes.
    """
    n = len(buf)
    if box_size is not None and box_size > 0:
        n_boxes = (n + box_size - 1) // box_size
        padded_size = _next_power_of_2(n_boxes) * box_size
    else:
        padded_size = _next_power_of_2(n)
    if padded_size == n:
        return buf
    return jnp.concatenate([buf, jnp.zeros(padded_size - n, dtype=buf.dtype)])


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n (minimum 1)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


CELL_TIERS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
              8192, 16384, 32768, 65536]


def _cell_tier(n_cells):
    """Return the smallest tier ceiling >= n_cells."""
    for t in CELL_TIERS:
        if n_cells <= t:
            return t
    return _next_power_of_2(n_cells)


def build_buckets(fb, dh, lev=0, max_boxes=None):
    """Group boxes by cell-count tier into BucketContext instances.

    Boxes with different (Nx, Ny, Nz) but similar cell counts are placed
    in the same bucket. The inner vmap size (n_cells_padded) is the tier
    ceiling. The outer vmap size (max_boxes) is power-of-2 padded.

    Returns list of BucketContext.
    """
    ng = fb.n_grow
    n_boxes = len(fb.offsets)

    # Group boxes by cell-count tier
    tier_groups = {}  # tier -> list of (mf_idx, offset, Nx, Ny, Nz, n_cells)
    for b in range(n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        vNz = Nz - 2 * ng
        n_cells = vNx * vNy * vNz
        tier = _cell_tier(n_cells)
        tier_groups.setdefault(tier, []).append(
            (b, int(fb.offsets[b]), Nx, Ny, Nz, n_cells)
        )

    result = []
    for tier, boxes in tier_groups.items():
        n_valid = len(boxes)
        mb = max_boxes if max_boxes is not None else _next_power_of_2(n_valid)

        offsets = [b[1] for b in boxes]
        indices = [b[0] for b in boxes]
        Nx_list = [b[2] for b in boxes]
        Ny_list = [b[3] for b in boxes]
        Nz_list = [b[4] for b in boxes]
        nc_list = [b[5] for b in boxes]

        # Pad to max_boxes — replicate first box's values for dummies
        dummy_off = offsets[0]
        dummy_Nx = Nx_list[0]
        dummy_Ny = Ny_list[0]
        dummy_Nz = Nz_list[0]
        dummy_nc = nc_list[0]

        pad_n = mb - n_valid
        offsets.extend([dummy_off] * pad_n)
        Nx_list.extend([dummy_Nx] * pad_n)
        Ny_list.extend([dummy_Ny] * pad_n)
        Nz_list.extend([dummy_Nz] * pad_n)
        nc_list.extend([dummy_nc] * pad_n)

        # dh is uniform per level, broadcast to per-box
        dh_row = list(dh)
        dh_data = [dh_row] * mb

        bucket = BucketContext(
            box_offsets=jnp.array(offsets[:mb], dtype=jnp.int32),
            cell_buf=fb.contiguous_array,
            Nx_arr=jnp.array(Nx_list[:mb], dtype=jnp.int32),
            Ny_arr=jnp.array(Ny_list[:mb], dtype=jnp.int32),
            Nz_arr=jnp.array(Nz_list[:mb], dtype=jnp.int32),
            n_cells_arr=jnp.array(nc_list[:mb], dtype=jnp.int32),
            dh_arr=jnp.array(dh_data, dtype=jnp.float64),
            ng=ng,
            n_cells_padded=tier,
            max_boxes=mb,
            n_valid=n_valid,
            box_indices=tuple(indices),
            lev=lev,
        )
        result.append(bucket)
    return result
