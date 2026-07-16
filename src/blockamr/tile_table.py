# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tile table for uniform-kernel dispatch over AMR boxes.

Each AMR box is tiled into bf×bf×bf chunks. The TileTable holds
per-tile metadata (offset, strides) into the MultiFab contiguous buffer.
All metadata are JAX arrays (data, not static) so that box count and
box sizes can change without recompilation.

Built from MultiFab.tile_table(bf) on the C++ side — pure metadata, no copies.
"""

import equinox as eqx
import jax.numpy as jnp


class TileTable(eqx.Module):
    """Flat table of tile descriptors into a contiguous MultiFab buffer.

    Each tile is a bf×bf×bf region of valid cells within a box.
    Strides are the parent box's Fortran-order strides, so indexing
    reads directly from the shared contiguous buffer.

    All array fields are (n_padded,) — padded to power-of-2 for vmap.
    Actual tile count is n_tiles.
    """

    offset: jnp.ndarray     # (n_padded,) start index into contiguous buffer
    stride_x: jnp.ndarray   # (n_padded,) = 1 (Fortran x-fastest)
    stride_y: jnp.ndarray   # (n_padded,) = Nx of parent box
    stride_z: jnp.ndarray   # (n_padded,) = Nx * Ny of parent box
    stride_c: jnp.ndarray   # (n_padded,) = Nx * Ny * Nz of parent box
    box_id: jnp.ndarray     # (n_padded,) which box this tile belongs to
    tile_i: jnp.ndarray     # (n_padded,) tile x-index within box
    tile_j: jnp.ndarray     # (n_padded,) tile y-index within box
    tile_k: jnp.ndarray     # (n_padded,) tile z-index within box
    n_tiles: int = eqx.field(static=True)   # actual tile count
    n_padded: int = eqx.field(static=True)  # padded count (power of 2)
    bf: int = eqx.field(static=True)        # blocking factor (tile size)
    ng: int = eqx.field(static=True)        # ghost cell width


def tile_table_from_multifab(mf, bf=4):
    """Build TileTable from a MultiFab (calls C++ tile_table method).

    Parameters
    ----------
    mf : MultiFab
        Must be allocated with single-chunk mode.
    bf : int
        Blocking factor (tile size per dimension).

    Returns
    -------
    TileTable
    """
    d = mf.tile_table(bf)
    return TileTable(
        offset=jnp.array(d["offset"]),
        stride_x=jnp.array(d["stride_x"]),
        stride_y=jnp.array(d["stride_y"]),
        stride_z=jnp.array(d["stride_z"]),
        stride_c=jnp.array(d["stride_c"]),
        box_id=jnp.array(d["box_id"]),
        tile_i=jnp.array(d["tile_i"]),
        tile_j=jnp.array(d["tile_j"]),
        tile_k=jnp.array(d["tile_k"]),
        n_tiles=int(d["n_tiles"]),
        n_padded=int(d["n_padded"]),
        bf=int(d["bf"]),
        ng=int(d["ng"]),
    )
