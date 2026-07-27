# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``ghostCell`` — the sharp-interface method: preprocessing and its data (B8).

The method itself is small, because everything method-*agnostic* already lives
on the mesh (``plans/IBM/design.md`` §2.1): the classification, the signed
distance, the wall normal and the body intercept. What is left — and what this
module owns — is the one extra the method's boundary schemes need:

    for every **fluid** wall-layer cell (``depth == 1``), a point on the
    outward normal a known distance from the surface, and the trilinear
    stencil that reads the field there.

That is :class:`GhostCellData`, declared *here*, next to the method, and stored
by ``mesh.ibm.data(GhostCell, lev)`` **opaquely** — the mesh never looks inside
it and neither does the operator, the backend or any other method (design
§2.4). The same object feeds all three of this method's boundary schemes; only
they read it.

The image point, and why it is where it is
------------------------------------------

For a fluid band cell ``P`` at signed distance ``s > 0`` from its patch, with
outward unit normal ``n̂``::

    x_i = x_P + h n̂        d_i = s + h

``h`` is the **largest step along n̂ that moves the image point at most half a
cell in every index direction**, ``h = 0.5 / max_d(|n̂_d| / dx_d)``. Two
properties of that choice are load-bearing:

* every trilinear donor is then within **one** cell of ``P`` in every
  direction, so a field with ``ngrow = 1`` already holds every value the row
  reads — the reach of the old mirror geometry (up to 7 cells on a quasi-2-D
  grid) is gone;
* ``d_i >= h >= 0.5 min(dx)`` bounds the wall closure's amplification. Closing
  against the cell centre itself (``d_i = s``) would be exact and interpolation
  free, and is what design §9 reads like at first — but ``s`` is *not* bounded
  away from zero (a cell centre can sit on the surface: measured ``s/dx ~
  1e-16`` on the tilted-plane rung), and the closure's ``1/d_i`` would then
  amplify round-off without limit.

``d_i`` is the distance **along the normal**, so a field that is linear along
``n̂`` is reproduced exactly: trilinear interpolation is linear-exact, and the
image point sits on the normal ray through ``x_P``.

Non-fluid cells get no image point at all. Under this design nothing is ever
reconstructed *inside* the body — a non-fluid cell is a row with ``nnz = 0``
(design §2.3) — so the mirror ``x + 2|s| n̂`` of the previous design, and the
enforcement ladder that kept its donors in the fluid, are both gone.
"""

from dataclasses import dataclass

import numpy as np

from .classify import _cell_name, _fluid_at_index, _index_coords, _patches, box_grids
from .geometry import _trilinear_donors

#: Trilinear stencil size — the donor slot count of :class:`GhostCellData`.
K = 8


@dataclass(frozen=True)
class GhostCellData:
    """``ghostCell``'s own extras. Nothing outside this method reads these.

    One entry per **fluid wall-layer cell** (``depth == 1``) of the level, in
    the order those cells appear per local box in ``MFIterator`` order — the
    same order :class:`~blockamr.ibm.band.Band` lists them in, so a boundary
    scheme selects them from its band with ``band.depth == 1`` and needs no
    lookup table.

    ``weight`` sums to one over the live slots. A slot whose weight is exactly
    zero is dead: its ``donor`` is the cell itself, so it is inside every bound
    and is never a non-fluid read.
    """

    image_point: np.ndarray  # f64   (n, 3)
    donor: np.ndarray  # int32 (n, K, 3), global index, unwrapped
    weight: np.ndarray  # f64   (n, K)
    distance: np.ndarray  # f64   (n,), image point to surface along n̂

    @property
    def nrows(self):
        """Number of fluid wall-layer cells on this level."""
        return int(self.image_point.shape[0])


class GhostCell:
    """Operator method: the wall condition enters each operator as its own
    boundary scheme, built on the image points this class precomputes."""

    name = "ghostCell"
    kind = "operator"
    requires_bodies = True
    data_type = GhostCellData

    @staticmethod
    def preprocess(mesh, lev):
        """Image point + interpolation stencil of every fluid wall-layer cell.

        Pure geometry: it never reads a field value, so it is cached per
        ``(method, lev, grid_version)`` and rebuilt only by a regrid or a moved
        body.
        """
        ibm = mesh.ibm
        names, body_list = _patches(ibm.bodies)
        return ghost_cell_data(box_grids(mesh, lev), ibm.geometry(lev), names, body_list)


def ghost_cell_data(grids, geometries, names, body_list):
    """:meth:`GhostCell.preprocess` on explicit per-box descriptions.

    ``grids`` and ``geometries`` are one per local box, in ``MFIterator``
    order; ``names``/``body_list`` are :func:`~blockamr.ibm.classify._patches`
    of the mesh's bodies.
    """
    blocks = [_box_data(grid, geom, names, body_list) for grid, geom in zip(grids, geometries)]
    if not blocks:
        blocks = [_empty()]
    return GhostCellData(
        image_point=_concat(blocks, "image_point", (0, 3), np.float64),
        donor=_concat(blocks, "donor", (0, K, 3), np.int32),
        weight=_concat(blocks, "weight", (0, K), np.float64),
        distance=_concat(blocks, "distance", (0,), np.float64),
    )


def image_step(normal, dx):
    """The step along ``n̂`` that moves the image at most half a cell per axis.

    ``0.5 / max_d(|n̂_d| / dx_d)`` — the largest such step, so the closure
    distance is as long (and the amplification as small) as one ghost layer
    allows.
    """
    reach = np.max(np.abs(normal) / np.asarray(dx, dtype=float), axis=-1)
    return 0.5 / reach


def _box_data(grid, geometry, names, body_list):
    """The image points of one box's fluid wall-layer cells."""
    selected = geometry.depth == 1
    cell = np.argwhere(selected) + np.asarray(grid.lo)
    if cell.shape[0] == 0:
        return _empty()

    normal = geometry.normal[selected]
    step = image_step(normal, grid.dx)
    image_point = _index_coords(cell, grid) + step[:, np.newaxis] * normal
    donor, weight = _trilinear_donors(image_point, grid)
    _check_fluid_donors(cell, donor, weight, geometry.patch[selected], names, body_list, grid)

    # A dead slot (weight exactly zero) is never read; pointing it at its own
    # cell keeps it inside every bound the handle checks and inside the fluid.
    live = weight != 0.0
    donor = np.where(live[..., np.newaxis], donor, cell[:, np.newaxis, :])
    return {
        "image_point": image_point,
        "donor": donor,
        "weight": weight,
        "distance": geometry.sdf[selected] + step,
    }


def _check_fluid_donors(cell, donor, weight, patch, names, body_list, grid):
    """Invariant F, at build time: every live donor is a fluid cell.

    The row builder is the only layer that can tell fluid from solid cheaply,
    and a violation is a wrong number rather than a crash — so it is checked
    here and not in the kernel (``plans/IBM/row-contract.md`` §8).
    """
    bad = ~_fluid_at_index(donor, grid, body_list) & (weight != 0.0)
    if not bad.any():
        return
    r, k = (int(v) for v in np.argwhere(bad)[0])
    raise ValueError(
        f"IBM band cell {_cell_name(cell[r])} on patch '{names[patch[r]]}' interpolates its "
        f"image point from {_cell_name(donor[r, k])}, which is not a fluid cell (Invariant F: "
        "a live stencil entry must be fluid, because a non-fluid cell holds the pin value and "
        "not data). The fluid on that side of the surface is under one cell deep there — "
        "refine the mesh or move the bodies apart."
    )


def _empty():
    return {
        "image_point": np.zeros((0, 3)),
        "donor": np.zeros((0, K, 3), dtype=np.int64),
        "weight": np.zeros((0, K)),
        "distance": np.zeros(0),
    }


def _concat(blocks, key, empty_shape, dtype):
    parts = [blk[key] for blk in blocks if blk[key].shape[0]]
    if not parts:
        return np.zeros(empty_shape, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(parts), dtype=dtype)
