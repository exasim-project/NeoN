# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Flat contiguous buffer data structures for bucketed dispatch."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class FlattenedBoxes(eqx.Module):
    """Contiguous MultiFab data with per-box and per-tile metadata.

    eqx.Module — shapes, n_grow, n_tiles_padded, bf are static
    (trigger recompile when changed). contiguous_array, offsets, tiles,
    n_tiles are traced leaves.
    """

    contiguous_array: Array  # (total_elems,) flat 1D buffer
    offsets: Array  # (n_boxes,) int — start offset of each box
    shapes: tuple = eqx.field(static=True)  # ((Nx,Ny,Nz,nc), ...) per box
    n_grow: int = eqx.field(static=True)  # ghost cell layers

    # Tile metadata — packed [offset, sx, sy, sz, box_id] per tile
    tiles: Array = None  # (n_tiles_padded * 5,) int32, traced
    n_tiles: Array = None  # int32 scalar, traced (for pl.when)
    n_tiles_padded: int = eqx.field(static=True, default=0)
    bf: int = eqx.field(static=True, default=0)

    @property
    def n_boxes(self):
        return len(self.offsets)


def flattened_boxes_from_mf(mf, bf=0, padded_n_elems=0):
    """Construct FlattenedBoxes from a MultiFab.

    Uses contiguous_array() (zero-copy) and fab_metadata() from the
    C++ bindings. When bf > 0, also builds packed tile metadata via
    the C++ packed_tiles() method.

    When padded_n_elems > 0, the contiguous_array view includes
    padding bytes so the shape stays stable across AMR regrids.
    """
    values = mf.contiguous_array(padded_n_elems)
    meta = mf.fab_metadata()
    offsets = jnp.array([m[0] for m in meta], dtype=jnp.int32)
    shapes = tuple((m[1], m[2], m[3], m[4]) for m in meta)

    tiles = n_tiles = None
    n_tiles_padded = bf_val = 0
    if bf > 0:
        d = mf.packed_tiles(bf)
        tiles = jnp.array(d["tiles"])
        n_tiles = jnp.array(int(d["n_tiles"]), dtype=jnp.int32)
        n_tiles_padded = int(d["n_padded"])
        bf_val = int(d["bf"])

    return FlattenedBoxes(
        contiguous_array=values, offsets=offsets, shapes=shapes,
        n_grow=mf.n_grow(),
        tiles=tiles, n_tiles=n_tiles, n_tiles_padded=n_tiles_padded, bf=bf_val,
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

    def replace_buf(self, new_buf):
        """Return a copy with a new cell_buf (avoids full rebuild)."""
        return eqx.tree_at(lambda s: s.cell_buf, self, new_buf)


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


class ElementMap(eqx.Module):
    """Maps flat element indices to box and local cell indices.

    Used by flat dispatch (process_flat / evaluate_flat) to iterate over
    all valid cells on a level with a single vmap + lax.fori_loop,
    eliminating per-bucket kernel launch overhead.
    """

    elem_to_box: Array       # (total_padded,) int32 — box index per element
    elem_to_cell_idx: Array  # (total_padded,) int32 — local cell_idx per element
    total_valid: int = eqx.field(static=True)   # actual valid element count
    total_padded: int = eqx.field(static=True)  # padded to chunk_size multiple
    chunk_size: int = eqx.field(static=True)    # elements per lax.fori_loop chunk


CELL_TIERS = [32, 64, 128, 256, 512, 1024, 2048, 4096,
              8192, 16384, 32768, 65536, 131072, 262144]


def _cell_tier(n_cells):
    """Return the smallest tier ceiling >= n_cells."""
    for t in CELL_TIERS:
        if n_cells <= t:
            return t
    return _next_power_of_2(n_cells)


MAX_BOXES_FIXED = 128  # starting outer vmap size — grows by 2x if exceeded


def _box_tier(n_boxes):
    """Return the smallest box tier >= n_boxes. Starts at 128, grows by 2x."""
    t = MAX_BOXES_FIXED
    while t < n_boxes:
        t *= 2
    return t


def build_fixed_buckets(fb, dh, lev=0):
    """Tier-bucketed cells + fixed MAX_BOXES — minimal recompilation.

    Groups boxes by cell-count tier (like build_buckets), but pads
    max_boxes to MAX_BOXES_FIXED. Recompiles only when a new cell
    tier appears (rare with power-of-2 tiers growing by 2x).

    Static fields per bucket: n_cells_padded (cell tier), max_boxes (box tier),
    n_valid (padded = max_boxes), box_indices (padded to max_boxes), ng, lev.
    Recompiles only when box count exceeds current box tier (grows 2x from 128)
    or a new cell tier appears.
    """
    ng = fb.n_grow
    n_boxes = len(fb.offsets)

    tier_groups = {}
    for b in range(n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx = Nx - 2*ng; vNy = Ny - 2*ng; vNz = Nz - 2*ng
        n_cells = vNx * vNy * vNz
        tier = _cell_tier(n_cells)
        tier_groups.setdefault(tier, []).append(
            (b, int(fb.offsets[b]), Nx, Ny, Nz, n_cells))

    result = []
    for tier, boxes in tier_groups.items():
        n_valid = len(boxes)
        mb = _box_tier(n_valid)

        offsets = [b[1] for b in boxes]
        indices = [b[0] for b in boxes]
        Nx_list = [b[2] for b in boxes]
        Ny_list = [b[3] for b in boxes]
        Nz_list = [b[4] for b in boxes]
        nc_list = [b[5] for b in boxes]

        # Pad to box tier — dummy boxes replicate first box's geometry
        # but have the same n_cells (compute runs, result is duplicate of box 0)
        pad_n = mb - n_valid
        if pad_n > 0:
            offsets += [offsets[0]] * pad_n
            Nx_list += [Nx_list[0]] * pad_n
            Ny_list += [Ny_list[0]] * pad_n
            Nz_list += [Nz_list[0]] * pad_n
            nc_list += [nc_list[0]] * pad_n
            indices += [indices[0]] * pad_n

        dh_data = [list(dh)] * mb
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
            n_valid=mb,  # padded — dummy boxes replicate first box
            box_indices=tuple(indices[:mb]),
            lev=lev,
        )
        result.append(bucket)
    return result


def build_buckets(fb, dh, lev=0, max_boxes=None, fixed_boxes=False):
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
        if fixed_boxes:
            mb = MAX_BOXES_FIXED
        elif max_boxes is not None:
            mb = max_boxes
        else:
            mb = _next_power_of_2(n_valid)

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

        # Pad box_indices to mb (dummy indices point to first box)
        padded_indices = indices + [indices[0]] * (mb - n_valid)

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
            box_indices=tuple(padded_indices[:mb]),
            lev=lev,
        )
        result.append(bucket)
    return result


TOTAL_TIERS = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
]


def _total_tier(n):
    """Return the smallest total tier >= n."""
    for t in TOTAL_TIERS:
        if n <= t:
            return t
    return _next_power_of_2(n)


def build_flat_context(fb, dh, lev=0, pad_strategy="power2"):
    """Build a BucketContext + ElementMap for flat element-level dispatch.

    All boxes on a level are placed in a single BucketContext.  The
    ElementMap maps flat valid-cell indices to (box_idx, cell_idx) pairs
    for use with ``process_flat`` / ``evaluate_flat``.

    Parameters
    ----------
    fb : FlattenedBoxes
        Per-level flat MultiFab data.
    dh : tuple of float
        Cell spacing (dx, dy, dz).
    lev : int
        AMR level index.
    pad_strategy : str
        "power2" — next power-of-2 (default, few recompiles)
        "tier"   — coarse tiers (TOTAL_TIERS), even fewer recompiles
        "fixed"  — pad to a large fixed max (zero recompiles, more waste)
    """
    ng = fb.n_grow
    n_boxes = fb.n_boxes

    offsets = []
    Nx_list = []
    Ny_list = []
    Nz_list = []
    nc_list = []
    elem_box = []
    elem_cell = []
    max_n_cells = 0

    for b in range(n_boxes):
        Nx, Ny, Nz = fb.shapes[b][:3]
        vNx = Nx - 2 * ng
        vNy = Ny - 2 * ng
        vNz = Nz - 2 * ng
        n_valid = vNx * vNy * vNz
        max_n_cells = max(max_n_cells, n_valid)

        offsets.append(int(fb.offsets[b]))
        Nx_list.append(Nx)
        Ny_list.append(Ny)
        Nz_list.append(Nz)
        nc_list.append(n_valid)

        elem_box.extend([b] * n_valid)
        elem_cell.extend(range(n_valid))

    total_valid = len(elem_box)
    if pad_strategy == "tier":
        total_padded = _total_tier(total_valid)
    elif pad_strategy == "fixed":
        total_padded = TOTAL_TIERS[-1]  # 4M — fits most AMR meshes
    else:  # "power2"
        total_padded = _next_power_of_2(total_valid)

    # Pad element arrays — dummy elements point at box 0, cell 0
    pad_n = total_padded - total_valid
    if pad_n > 0:
        elem_box.extend([0] * pad_n)
        elem_cell.extend([0] * pad_n)

    # Pad box arrays to power-of-2
    mb = _next_power_of_2(n_boxes)
    pad_boxes = mb - n_boxes
    if pad_boxes > 0:
        offsets.extend([offsets[0]] * pad_boxes)
        Nx_list.extend([Nx_list[0]] * pad_boxes)
        Ny_list.extend([Ny_list[0]] * pad_boxes)
        Nz_list.extend([Nz_list[0]] * pad_boxes)
        nc_list.extend([nc_list[0]] * pad_boxes)

    dh_data = [list(dh)] * mb
    max_tier = _cell_tier(max_n_cells)

    bucket = BucketContext(
        box_offsets=jnp.array(offsets[:mb], dtype=jnp.int32),
        cell_buf=fb.contiguous_array,
        Nx_arr=jnp.array(Nx_list[:mb], dtype=jnp.int32),
        Ny_arr=jnp.array(Ny_list[:mb], dtype=jnp.int32),
        Nz_arr=jnp.array(Nz_list[:mb], dtype=jnp.int32),
        n_cells_arr=jnp.array(nc_list[:mb], dtype=jnp.int32),
        dh_arr=jnp.array(dh_data, dtype=jnp.float64),
        ng=ng,
        n_cells_padded=max_tier,
        max_boxes=mb,
        n_valid=n_boxes,
        box_indices=tuple(range(n_boxes)),
        lev=lev,
    )

    elem_map = ElementMap(
        elem_to_box=jnp.array(elem_box[:total_padded], dtype=jnp.int32),
        elem_to_cell_idx=jnp.array(elem_cell[:total_padded], dtype=jnp.int32),
        total_valid=total_valid,
        total_padded=total_padded,
        chunk_size=total_padded,  # single chunk = full vmap
    )

    return bucket, elem_map
