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
    _index_grid,
    _owner,
    _patches,
    _sdf_stack,
    _valid_index,
    box_grids,
    classify_box,
)

#: The eight trilinear corner offsets, in donor-slot order.
_OFFSETS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)

# ---------------------------------------------------------------------------
# the v2 packed geometry (B29) — additive; nothing above changes
# ---------------------------------------------------------------------------

#: Component offsets of the packed v2 geometry fab. These mirror ``ibm::GEOM_*``
#: in ``src/bindings/blockAMR/ibm/geometry_view.H``, and since B31 the compiled
#: side exports all five (``blockamr.GEOM_SDF`` … ``blockamr.IBM_GEOM_NCOMP``)
#: and ``test_ibm_mesh.py``'s layout row asserts every one of them against these
#: names. The offsets are therefore **pinned** across the language boundary, not
#: merely mirrored: the layout is named in exactly two places and they cannot
#: drift.
GEOM_SDF = 0
GEOM_NORMAL = 1
GEOM_WALL_POINT = 4
GEOM_PATCH = 7
GEOM_NCOMP = 8


def packed_box_geometry(grid, body_list, ngrow):
    """The packed 8-component v2 geometry of one box, GROWN by ``ngrow``.

    The Fortran-ordered ``(nx, ny, nz, 8)`` block that
    :meth:`~blockamr.ibm.mesh.IbmMesh.geometry_fab` uploads with
    ``MultiFab.copy_grown_from`` (review.md §4, Q29(d): the v2 geometry is
    *uploaded* from this numpy evaluation, so B31's parity bar tests its own
    arithmetic and not a second geometry implementation).

    **The ghost contract (review F10), honoured here.** The evaluation runs on
    the valid box grown by ``ngrow`` — never read back from a MultiFab and never
    filled by ``FillBoundary`` — and it takes coordinates from
    :func:`~blockamr.ibm.classify._index_coords`, which **wraps** in a periodic
    direction and extends in a non-periodic one. Both halves are load-bearing:

    * across a periodic seam the ghost must carry the *wrapped* cell's geometry,
      because ``classify_default`` fills the marker's ghosts by ``FillBoundary``
      from that same wrapped valid cell and the always-on M5 check compares the
      two at one index. Evaluating the body at the unwrapped coordinate makes
      every classification of a body near a periodic boundary throw M5.
    * outside a non-periodic domain face there is no neighbour to copy from at
      all, so only an analytic evaluation puts a meaningful value there — which
      is what makes the classification's first pass right on *that* shell, the
      one no exchange could have filled. Elsewhere (across a box edge, across a
      periodic seam) the marker's ghosts come from ``classify_default``'s two
      ``FillBoundary`` calls; a ghost ``WALL`` in particular can only have
      arrived that way, since pass 2 launches on the valid box.

    ``wall_point`` therefore also uses the wrapped position: a ghost cell *is*
    the wrapped cell, geometry and all. That is the opposite convention from
    :func:`_donor_coords`, which describes an interpolation stencil rather than
    a cell.

    Unlike :func:`box_geometry` this computes **no** ``depth`` (design §2.1: the
    v2 marker answers what depth answered), and it runs none of v1's band-time
    validity checks — ``classify_box``'s ``_check_adjacent`` and
    ``_check_resolvable_gap`` are statements about the v1 band, and the v2 path's
    conformance checks are M4/M5 in ``validate_cell_type``. Whether the thin-gap
    check gets a v2 home is B36's to decide.
    """
    coords = _index_coords(_index_grid(grid, int(ngrow)), grid)
    out = np.zeros(coords.shape[:-1] + (GEOM_NCOMP,), dtype=float)
    if not body_list:
        # no bodies: every cell is fluid, exactly as ``classify_box`` reports it
        out[..., GEOM_SDF] = np.inf
        out[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3] = coords
        return np.asfortranarray(out)

    s_all = _sdf_stack(body_list, coords[..., 0], coords[..., 1], coords[..., 2])
    owner, _s_owner = _owner(s_all)
    sdf = s_all.min(axis=0)
    normal = _normals(coords.reshape(-1, 3), owner.ravel(), body_list).reshape(coords.shape)

    out[..., GEOM_SDF] = sdf
    out[..., GEOM_NORMAL : GEOM_NORMAL + 3] = normal
    # the same intercept ``box_geometry`` builds, from the same union sdf
    out[..., GEOM_WALL_POINT : GEOM_WALL_POINT + 3] = coords - sdf[..., np.newaxis] * normal
    out[..., GEOM_PATCH] = owner
    return np.asfortranarray(out)


def packed_geometry_on_grids(grids, bodies, ngrow):
    """:func:`packed_box_geometry` per box, in ``MFIterator`` order."""
    # patch ids are indices into ``sorted(bodies)`` — the same order v1 uses
    _names, body_list = _patches(bodies)
    return [packed_box_geometry(grid, body_list, ngrow) for grid in grids]


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
