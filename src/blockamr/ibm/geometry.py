# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Per-cell wall geometry — the method-agnostic IBM layer (task B1).

:class:`IbmGeometry` is what the mesh stores and what every boundary scheme
reads (``plans/IBM/design.md`` §2.1): the classification from
:mod:`.classify` plus the wall geometry each band cell needs — the union
signed distance, the owning body's unit normal and the body intercept.

Method-specific extras (``ghostCell``'s image points, a cut-cell's face areas)
are **not** here: a method declares and registers its own data type next to
itself, and nothing outside that method reads it (design §2.4).

Pure numpy on explicit per-box index ranges, exactly like :mod:`.classify`, so
the whole layer is testable without the compiled extension; the only entry
point that touches ``blockamr`` is :func:`ibm_geometry`, through
:func:`~blockamr.ibm.classify.box_grids`.

Interpolation geometry uses the **unwrapped** cell position
(:func:`_donor_coords`), while the fluid/solid state of the same index is
judged at the wrapped one — the two conventions :mod:`.classify` documents.
"""

from dataclasses import dataclass

import numpy as np

from .classify import (
    _index_coords,
    _patches,
    _valid_index,
    box_grids,
    classify_box,
)

#: The eight trilinear corner offsets, in donor-slot order.
_OFFSETS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


@dataclass(frozen=True)
class IbmGeometry:
    """Cell-centred wall geometry of one box, method-agnostic (design §2.1).

    Built once per grid generation from ``mesh.bodies`` alone, and read by the
    band builder and by every boundary scheme::

        geom = ibm_geometry(mesh, lev, mesh.bodies)[box]
        band = np.argwhere(geom.depth <= width)

    ``sdf``, ``normal`` and ``wall_point`` are stored full-field even though
    only the band needs them: they are cheap analytic evaluations, and a full
    field removes an indirection from every consumer. ``wall_point`` is only
    meaningful for ``|depth| <= 1``, and ``non_fluid_pin`` is the value
    preprocessing pins non-fluid cells to so the interior sweep's discarded
    reads are finite (design §7).
    """

    depth: np.ndarray  # int8   (nx, ny, nz)
    patch: np.ndarray  # int8   (nx, ny, nz)
    sdf: np.ndarray  # f64    (nx, ny, nz)
    normal: np.ndarray  # f64    (nx, ny, nz, 3)
    wall_point: np.ndarray  # f64    (nx, ny, nz, 3)
    non_fluid_pin: float = 0.0


def ibm_geometry(mesh, lev, bodies):
    """:class:`IbmGeometry` per local box of ``lev``, in ``MFIterator`` order.

    ``bodies`` is the patch-keyed ``mesh.bodies`` dict; patch ids are indices
    into ``sorted(bodies)``.
    """
    return geometry_on_grids(box_grids(mesh, lev), bodies)


def geometry_on_grids(grids, bodies):
    """:func:`ibm_geometry` on explicit :class:`BoxGrid` descriptions."""
    names, body_list = _patches(bodies)
    return [box_geometry(grid, names, body_list) for grid in grids]


def box_geometry(grid, names, body_list):
    """The wall geometry of one box's valid cells."""
    depth, patch, sdf = classify_box(grid, names, body_list)
    coords = _index_coords(_valid_index(grid), grid)
    if not body_list:
        return IbmGeometry(
            depth=depth,
            patch=patch,
            sdf=sdf,
            normal=np.zeros(coords.shape),
            wall_point=coords,
        )
    normal = _normals(coords.reshape(-1, 3), patch.ravel(), body_list).reshape(coords.shape)
    return IbmGeometry(
        depth=depth,
        patch=patch,
        sdf=sdf,
        normal=normal,
        # the intercept along the owning body's normal; in the overlap of two
        # bodies ``sdf`` is the deeper one's, which is why it is documented as
        # meaningful only next to the surface (|depth| <= 1).
        wall_point=coords - sdf[..., np.newaxis] * normal,
    )


def _normals(points, owner, body_list):
    """Per-point unit normal of the owning body, shape ``(n, 3)``."""
    out = np.zeros((points.shape[0], 3), dtype=float)
    for b, body in enumerate(body_list):
        sel = owner == b
        if sel.any():
            p = points[sel]
            out[sel] = body.normal(p[:, 0], p[:, 1], p[:, 2])
    return out


def _trilinear_donors(points, grid):
    """The 8 cells surrounding each point and their trilinear weights."""
    plo = np.asarray(grid.prob_lo)
    dx = np.asarray(grid.dx)
    t = (np.asarray(points, dtype=float) - plo) / dx - 0.5
    base = np.floor(t).astype(np.int64)
    frac = t - base
    idx = base[:, np.newaxis, :] + _OFFSETS[np.newaxis]
    corner = np.where(
        _OFFSETS[np.newaxis] == 0, 1.0 - frac[:, np.newaxis, :], frac[:, np.newaxis, :]
    )
    return idx, corner.prod(axis=2)


def _donor_coords(idx, grid):
    """Cell centres of ``idx``, **unwrapped**.

    This is the position the trilinear stencil was built around, so it is what
    the interpolation geometry must use.
    :func:`~blockamr.ibm.classify._index_coords` wraps instead, which is what
    the fluid/solid lookup wants (that halo cell *is* the wrapped cell) and
    what the geometry must not do.
    """
    return np.asarray(grid.prob_lo) + (idx + 0.5) * np.asarray(grid.dx)


def _containing_cell(points, grid):
    """The cell each point lies in (not the same as the donor base index)."""
    plo = np.asarray(grid.prob_lo)
    dx = np.asarray(grid.dx)
    return np.floor((np.asarray(points, dtype=float) - plo) / dx).astype(np.int64)
