# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The band — the boundary-cell set of a scheme of one stencil width (task B2).

``band(w) = {depth <= w}``: exactly the cells whose width-``w`` cross stencil is
not entirely in the fluid (``plans/IBM/overview.md`` §4). Everything else is
*bulk*, where the interior scheme's result is bitwise the no-body result, which
is why the whole IBM correction is a list of rows over this set and nothing
else.

Derived from :mod:`.classify`'s ``depth`` by a threshold, so one classification
serves every stencil width; the mesh caches the result per
``(lev, width, shape, grid_version)`` (design §8) and reaches it as
``mesh.ibm.band(lev, width)``.

Rows are grouped **per local box in ``MFIterator`` order** and addressed by the
CSR-style ``box_offset`` — the convention :mod:`.rows` already uses, kept
verbatim so a band row and a wall row are laid out the same way.
"""

from dataclasses import dataclass

import numpy as np

from .classify import MAX_DEPTH

#: Axis-ray stencil: the shape ``depth`` is defined against.
CROSS = "cross"

#: Corner-reading stencil; needs the Chebyshev depth, which is not built yet.
BOX = "box"

SHAPES = (BOX, CROSS)


@dataclass(frozen=True)
class Band:
    """The boundary cells of a width-``width`` scheme, one flat row list.

    Built by :func:`band_on_grids` and read by every boundary scheme::

        band = mesh.ibm.band(lev, width=1)
        rows_of_box_3 = slice(band.box_offset[3], band.box_offset[4])

    ``cell`` is in **global** index space (the same space :mod:`.rows` uses for
    its targets), so a row is addressable without knowing which box it came
    from. ``depth <= 0`` marks the non-fluid part of the band, where the
    operator's value is meaningless and the scheme emits an empty row.
    """

    width: int
    shape: str  # "cross" | "box"
    cell: np.ndarray  # int32 (n, 3), global index
    depth: np.ndarray  # int8  (n,)
    patch: np.ndarray  # int8  (n,)
    box_offset: np.ndarray  # int32 (nbox + 1,), CSR, MFIterator order

    @property
    def nrows(self):
        """Number of band cells on this level."""
        return int(self.cell.shape[0])


def band_on_grids(grids, geometries, width, shape=CROSS):
    """The :class:`Band` of ``width`` over the level's boxes.

    ``grids`` and ``geometries`` are the ``BoxGrid`` and
    :class:`~blockamr.ibm.geometry.IbmGeometry` of each local box, both in
    ``MFIterator`` order.
    """
    _check_shape(shape)
    _check_width(width)
    cells, depths, patches, counts = [], [], [], []
    for grid, geometry in zip(grids, geometries):
        selected = geometry.depth <= width
        cells.append(np.argwhere(selected) + np.asarray(grid.lo))
        depths.append(geometry.depth[selected])
        patches.append(geometry.patch[selected])
        counts.append(int(selected.sum()))
    return Band(
        width=int(width),
        shape=shape,
        cell=_concat(cells, (0, 3), np.int32),
        depth=_concat(depths, (0,), np.int8),
        patch=_concat(patches, (0,), np.int8),
        box_offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
    )


def _concat(blocks, empty_shape, dtype):
    if not blocks:
        return np.zeros(empty_shape, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=dtype)


def _check_shape(shape):
    if shape == CROSS:
        return
    if shape == BOX:
        raise NotImplementedError(
            f"stencil shape '{BOX}' needs the Chebyshev depth, which is built with the "
            f"first corner-reading scheme; the shapes are {list(SHAPES)}."
        )
    raise ValueError(f"unknown stencil shape '{shape}'; the shapes are {list(SHAPES)}.")


def _check_width(width):
    """``depth`` is clamped at ``MAX_DEPTH``, so a wider band is not expressible.

    Beyond the clamp the threshold would sweep in every cell that is merely
    "no state change within reach" and call it a boundary cell.
    """
    if width >= MAX_DEPTH:
        raise ValueError(
            f"band width {width} is not below the depth clamp MAX_DEPTH={MAX_DEPTH}: "
            "cells further than the clamp from a body are indistinguishable from cells "
            "at the clamp, so the band of that width cannot be read off depth."
        )
