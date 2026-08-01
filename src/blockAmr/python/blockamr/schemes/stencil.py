# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Stencil slicing primitives: pure functions that take GHOSTED JAX arrays and return
slices into them. Inlined by `jax.jit` at trace time, so no runtime overhead.
"""
from __future__ import annotations

from jax import Array


def S(u: Array, k: int, ax: int) -> Array:
    """Stencil slice along axis *ax*: k = -1 left, 0 centre, +1 right.

    Trims one cell from each end of *ax* (total trim 2).
    """
    start: int = k + 1
    sl: list[slice] = [slice(None)] * u.ndim
    sl[ax] = slice(start, start + u.shape[ax] - 2)
    return u[tuple(sl)]


def S_wide(u: Array, k: int, ax: int, width: int) -> Array:
    """Wider stencil slice: *width* is the half-width, trimmed from each end, and k
    ranges over -width..+width.
    """
    start: int = k + width
    trim: int = 2 * width
    sl: list[slice] = [slice(None)] * u.ndim
    sl[ax] = slice(start, start + u.shape[ax] - trim)
    return u[tuple(sl)]


def face(f: Array, side: int, ax: int) -> Array:
    """Left (*side*=0) or right (*side*=1) face of each cell.

    *f* is face-centred, one element longer than the cell count along *ax*; the result
    matches the cell count.
    """
    sl: list[slice] = [slice(None)] * f.ndim
    sl[ax] = slice(side, side + f.shape[ax] - 1)
    return f[tuple(sl)]


def interior(v: Array, skip_ax: int, width: int = 1) -> Array:
    """Trim *v* to interior cells on every axis but *skip_ax*, *width* cells per end."""
    for ax in range(v.ndim):
        if ax != skip_ax:
            v = S_wide(v, 0, ax, width)
    return v
