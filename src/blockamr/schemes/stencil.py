# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Stencil slicing primitives for structured-grid operators.

These are pure functions on JAX arrays. They are inlined by `jax.jit` at trace
time — no runtime overhead. All functions operate on arrays that include ghost
cells and return views (slices) into them.
"""
from __future__ import annotations

from jax import Array


def S(u: Array, k: int, ax: int) -> Array:
    """Stencil slice along axis *ax*.

    k = -1 → left neighbour, 0 → centre, +1 → right neighbour.
    Trims one cell from each end of axis *ax* (total trim = 2).
    """
    start: int = k + 1
    sl: list[slice] = [slice(None)] * u.ndim
    sl[ax] = slice(start, start + u.shape[ax] - 2)
    return u[tuple(sl)]


def S_wide(u: Array, k: int, ax: int, width: int) -> Array:
    """Stencil slice for a wider stencil.

    *width* is the half-width: trims *width* cells from each end.
    k ranges from -width to +width.
    """
    start: int = k + width
    trim: int = 2 * width
    sl: list[slice] = [slice(None)] * u.ndim
    sl[ax] = slice(start, start + u.shape[ax] - trim)
    return u[tuple(sl)]


def face(f: Array, side: int, ax: int) -> Array:
    """Extract the left (*side*=0) or right (*side*=1) face of each cell.

    *f* is a face-centred array with one more element than cells along *ax*.
    The result has the same number of elements as cells along *ax*.
    """
    sl: list[slice] = [slice(None)] * f.ndim
    sl[ax] = slice(side, side + f.shape[ax] - 1)
    return f[tuple(sl)]


def interior(v: Array, skip_ax: int, width: int = 1) -> Array:
    """Trim *v* to interior cells along every axis except *skip_ax*.

    *width* is the number of cells to trim from each end (matches stencil half-width).
    """
    for ax in range(v.ndim):
        if ax != skip_ax:
            v = S_wide(v, 0, ax, width)
    return v
