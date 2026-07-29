# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""An STL triangulation as an immersed body (API doc §6).

:class:`Stl` is the geometric peer of :class:`~blockamr.ibm.body.Cylinder`: it
exposes the same two primitives — ``sdf(x, y, z)`` positive in the fluid, and
``normal(x, y, z)`` pointing into the fluid — so it plugs into ``mesh.bodies``
wherever an analytic body does, and every layer above it (the classification,
the packed geometry, the marker, the ghost-cell method) is unchanged::

    mesh.bodies = {"hull": Stl("hull.stl", scale=0.001)}

The values come from AMReX's ``STLtools`` through
``blockamr.StlSurface`` — the same reader, BVH and exact point-to-triangle
distance the EB machinery uses — and **nothing here interpolates**.

Where the values are asked for
------------------------------

``STLtools`` fills a grid, not a point list: its only query is "signed distance
at ``origin + (i + 1/2) * dx``". That is not a restriction on this pipeline,
because every point the geometry core ever evaluates a body at lies on a
regular axis-aligned lattice — the level's cell centres
(:func:`~blockamr.ibm.classify._index_coords`), those centres grown by up to
``MAX_DEPTH`` ghosts, and the same centres shifted half a cell onto a face in
:func:`~blockamr.ibm.classify._check_resolvable_gap`. So :meth:`Stl.sdf`
recovers the lattice from the coordinates it is handed, fills it once, and
reads the answer off it. Each returned value is the exact distance **at that
coordinate**, not a value interpolated from a nearby sample.

A query that is *not* on a regular lattice cannot be served this way and is
**refused** with :class:`ValueError` rather than approximated — a silently
interpolated signed distance would degrade the wall geometry the ghost-cell
method reconstructs from.

The one derived quantity
------------------------

``STLtools`` exposes no surface normal, so :meth:`normal` is the gradient of
the signed distance by a central difference over a step ``1e-3`` of the lattice
spacing. For a true signed distance the gradient is exactly the unit normal and
the difference is *exact* wherever the nearest facet does not change across the
step, which is everywhere except the surface's medial set — the diagonal
bisectors of a box, the axis of a cylinder. There a signed distance has no
gradient at all and the difference returns the mean of the neighbouring facet
normals, which is the honest answer to an ill-posed question. It costs six
extra fills per evaluation of a level's geometry.

Caveats
-------

The triangulation is assumed **watertight**. An open shell has no inside, and
``STLtools`` decides inside/outside by counting ray crossings, so a shell
returns a sign that flips arbitrarily across the hole. There is no cheap check
for it and none is faked here: check the mesh in the tool that produced it.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Relative slack when deciding whether coordinates lie on a regular lattice.
#: They are built by :func:`~blockamr.ibm.classify._index_coords` as
#: ``prob_lo + (i + 1/2) * dx``, so the spacing is exact to a few ulp; this is
#: loose enough for that and far too tight to accept a scattered point set.
_LATTICE_TOL = 1e-9

#: Refuse to fill more than this many samples for one query. A lattice is
#: inferred from the *smallest* gap between distinct coordinates, so a nearly
#: coincident pair of points would otherwise ask for an unbounded block.
_MAX_SAMPLES = 1 << 24

#: Central-difference step for :meth:`Stl.normal`, as a fraction of the lattice
#: spacing. Small enough that it stays inside one facet's Voronoi cell, large
#: enough that the difference keeps ~12 significant digits.
_GRADIENT_FRACTION = 1e-3


def _axis_lattice(values):
    """``(first, step, count, index)`` of one axis of coordinates.

    ``values`` is a flat array of coordinates along one axis; the lattice is
    ``first + k * step`` for ``k`` in ``[0, count)``, and ``index`` is each
    value's ``k``. ``step`` is ``None`` when there is only one distinct value —
    the caller supplies a spacing then, since a single sample constrains none.
    """
    distinct = np.unique(values)
    index_of = np.zeros(values.shape, dtype=np.intp)
    if distinct.size == 1:
        return float(distinct[0]), None, 1, index_of

    gaps = np.diff(distinct)
    step = float(gaps.min())
    multiples = np.rint(gaps / step)
    if not np.all(np.abs(gaps - multiples * step) <= _LATTICE_TOL * step):
        raise ValueError(
            "Stl can only be evaluated on a regular axis-aligned lattice, and these "
            "coordinates are not on one: the gaps between distinct values are not "
            f"multiples of the smallest gap {step:.6g}. An STL body's signed distance "
            "comes from a grid fill, and interpolating it to an arbitrary point would "
            "degrade the wall geometry."
        )
    count = round(float(distinct[-1] - distinct[0]) / step) + 1
    index_of = np.rint((values - distinct[0]) / step).astype(np.intp)
    return float(distinct[0]), step, count, index_of


class _Lattice:
    """The block ``STLtools`` has to fill to answer one query.

    ``origin`` is the block's *low corner*, i.e. half a spacing below the first
    sample, because ``STLtools``' sample points are ``origin + (i + 1/2) * dx``.
    ``index`` is the block index of each queried point, and ``spacing`` is the
    spacing the coordinates themselves determined — ``None`` when they hold a
    single distinct value per axis and therefore determine none.
    """

    def __init__(self, origin, dx, count, index, shape, spacing):
        self.origin = origin
        self.dx = dx
        self.count = count
        self.index = index
        self.shape = shape
        self.spacing = spacing


def _lattice(x, y, z):
    """The :class:`_Lattice` a broadcast coordinate triple lies on."""
    coords = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    )
    shape = coords[0].shape
    axes = [_axis_lattice(c.ravel()) for c in coords]

    known = [step for _first, step, _count, _index in axes if step is not None]
    # A query with one distinct value per axis names one point and no spacing.
    # Any spacing samples it exactly, since the sample is the block's centre.
    spacing = min(known) if known else None
    fallback = spacing if spacing is not None else 1.0

    dx = tuple(fallback if step is None else step for _first, step, _count, _index in axes)
    count = tuple(c for _first, _step, c, _index in axes)
    origin = tuple(first - 0.5 * d for (first, _step, _count, _index), d in zip(axes, dx))
    index = tuple(i for _first, _step, _count, i in axes)

    samples = count[0] * count[1] * count[2]
    if samples > _MAX_SAMPLES:
        raise ValueError(
            f"Stl would have to fill {samples} lattice samples {count} to answer a query "
            f"about {coords[0].size} points; the points are on a lattice far finer than "
            "they populate. Evaluate the body on the mesh's own cell centres."
        )
    return _Lattice(origin, dx, count, index, shape, spacing)


@dataclass
class Stl:
    """An immersed body whose surface is a triangulation on disk.

    ``scale`` and ``center`` are AMReX's: every vertex of the file becomes
    ``v * scale + center``, so ``center`` **translates** the (scaled) body and
    is not a point it is centred on. ``reverse_normal`` flips the facet
    winding, which swaps which side of the surface is solid — the fix for a
    file whose normals point inward.

    The file is read once, on first use, and the reader is kept for the
    lifetime of the body: it rides the ``(method, lev, grid_version)`` caches in
    :class:`~blockamr.ibm.mesh.IbmMesh` exactly as an analytic body does, so an
    ``evaluate`` never re-reads it and never re-fills.
    """

    path: str
    scale: float = 1.0
    center: Sequence[float] = (0.0, 0.0, 0.0)
    reverse_normal: bool = False

    def __post_init__(self):
        # Checked eagerly and in pure Python: the compiled reader needs an
        # initialised AMReX, and a typo in a path should not wait for the first
        # classification to be reported.
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                f"Stl: no STL file at {self.path!r} (mesh.bodies entries are read from disk "
                "on first use)"
            )
        self._surface = None

    @property
    def surface(self):
        """The compiled ``blockamr.StlSurface``, read on first use."""
        if self._surface is None:
            import blockamr

            self._surface = blockamr.StlSurface(
                path=str(self.path),
                scale=float(self.scale),
                center=tuple(float(v) for v in self.center),
                reverse_normal=bool(self.reverse_normal),
            )
        return self._surface

    def _fill(self, lattice, shift=None):
        """One compiled fill of ``lattice``, read back at the queried points.

        ``shift`` displaces every sample by the same vector, which leaves the
        lattice a lattice — that is what makes :meth:`normal`'s six differences
        six ordinary fills.
        """
        origin = lattice.origin
        if shift is not None:
            origin = tuple(o + s for o, s in zip(origin, shift))
        block = self.surface.signed_distance_block(origin=origin, dx=lattice.dx, n=lattice.count)
        return block[lattice.index].reshape(lattice.shape)

    def sdf(self, x, y, z):
        """Signed distance to the triangulation: positive outside it (fluid)."""
        return self._fill(_lattice(x, y, z))

    def normal(self, x, y, z):
        """Outward (into-fluid) unit normal, shape ``(..., 3)``.

        The central difference of the signed distance — see the module
        docstring for why that is exact off the medial set, and what it returns
        on it. The step is a fraction of the *queried* lattice's spacing, so a
        query that names no spacing (one point) is refused rather than
        differenced over a length that has nothing to do with the body.
        """
        lattice = _lattice(x, y, z)
        if lattice.spacing is None:
            raise ValueError(
                "Stl.normal needs more than one point: it differences the signed distance "
                "over a fraction of the spacing between the queried points, and a single "
                "point names no spacing. Evaluate it on the cell centres of a box, which "
                "is what the IBM geometry does."
            )
        step = _GRADIENT_FRACTION * lattice.spacing

        gradient = np.empty(lattice.shape + (3,), dtype=float)
        for d in range(3):
            ahead = [0.0, 0.0, 0.0]
            ahead[d] = step
            behind = [0.0, 0.0, 0.0]
            behind[d] = -step
            forward = self._fill(lattice, ahead)
            backward = self._fill(lattice, behind)
            gradient[..., d] = (forward - backward) / (2.0 * step)

        length = np.linalg.norm(gradient, axis=-1)
        # A zero gradient is only reachable on the medial set of a symmetric
        # body; leaving the vector at zero keeps downstream arithmetic finite,
        # exactly as Cylinder.normal does on its axis.
        return gradient / np.where(length > 0.0, length, 1.0)[..., np.newaxis]
