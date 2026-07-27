# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``mesh.ibm`` — the shared interface between preprocessing and the schemes (B2).

Everything the IBM precomputes from ``mesh.bodies`` hangs here: the per-cell
:class:`~blockamr.ibm.geometry.IbmGeometry`, the :class:`~blockamr.ibm.band.Band`
of a given stencil width, and each method's own data. Nothing is built until it
is asked for, and every cache is keyed by the grid generation (design §8), so a
regrid or a moved body cannot be served from a stale entry::

    mesh.bodies = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}
    band = mesh.ibm.band(lev, width=1)         # classifies on first use
    donors = mesh.ibm.data("ghostCell", lev)   # the method's own extras

Method data is stored **opaquely**: the method declares the type, the mesh
allocates and invalidates it and never looks inside (design §2.4).
"""

import weakref

from .band import CROSS, band_on_grids
from .band_rows import band_table, pin_rows
from .classify import box_grids
from .geometry import geometry_on_grids


class IbmMesh:
    """The lazy IBM cache of one mesh, reached as ``mesh.ibm``.

    Constructed by the mesh, one per mesh, and never by a caller.
    ``grid_version`` is the generation everything here is keyed on; it is
    bumped by a regrid *and* by re-assigning ``mesh.bodies``, since a moved body
    invalidates exactly what a moved grid does. Mutating the ``bodies`` dict in
    place does not — re-assign it, or call :meth:`invalidate`.
    """

    def __init__(self, mesh):
        self._mesh = mesh
        self._grids = {}
        self._geometry = {}
        self._bands = {}
        self._method_data = {}
        self._pin_tables = {}
        self._pinned = weakref.WeakKeyDictionary()

    @property
    def mesh(self):
        """The mesh this belongs to — the schemes read ``geom(lev)`` off it."""
        return self._mesh

    @property
    def grid_version(self):
        """The generation every cache here is keyed on."""
        return self._mesh._ibm_generation

    @property
    def bodies(self):
        """The patch-keyed bodies the geometry is built from."""
        return self._mesh.bodies

    def geometry(self, lev):
        """Per-cell wall geometry of ``lev``, one entry per local box.

        In ``MFIterator`` order, so entry ``i`` is box ``i`` of every other
        per-box list on this level.
        """
        key = (lev, self.grid_version)
        if key not in self._geometry:
            self._geometry[key] = geometry_on_grids(self._boxes(lev), self.bodies)
        return self._geometry[key]

    def band(self, lev, width, shape=CROSS):
        """The boundary cells of a width-``width`` scheme on ``lev``."""
        key = (lev, int(width), shape, self.grid_version)
        if key not in self._bands:
            self._bands[key] = band_on_grids(self._boxes(lev), self.geometry(lev), width, shape)
        return self._bands[key]

    def data(self, method, lev):
        """The method's own preprocessed data, built once per generation.

        The result is whatever the method's ``preprocess`` returned and is
        stored as-is; nothing here reads a field of it.
        """
        key = (method, lev, self.grid_version)
        if key not in self._method_data:
            self._method_data[key] = method.preprocess(self._mesh, lev)
        return self._method_data[key]

    def ensure_pinned(self, field, method, lev):
        """Pin ``field`` on ``lev`` once per classification (B25, design §7).

        The pin belongs to the classification, not to the evaluate (Q3,
        ``plans/IBM/review.md`` §4): it is applied the first time a field meets
        a given ``(method, lev, grid_version)``, and every evaluate after that
        is a pure read — even of a solid cell someone dirtied in between. A new
        generation re-pins, along with everything else this cache holds.

        The **field** is part of the key on top of the ``(method, lev,
        grid_version)`` triple the design names, because the write lands in
        *that* field's storage and v1's classification never sees a field: a
        second field on the same generation is pinned on its own first
        evaluate. ``method`` is the strategy class :meth:`data` is keyed on, so
        both caches share one key shape.
        """
        key = (method, int(lev), self.grid_version)
        seen = self._pinned.setdefault(field, set())
        if key in seen:
            return
        self.pin_non_fluid(field, lev)
        seen.add(key)

    def pin_non_fluid(self, field, lev):
        """Write the pin value into every non-fluid cell of ``field`` (B7).

        The one write this design makes to a user field, and design §7 is where
        it is argued: the interior sweep reads non-fluid neighbours at a band
        cell, and those reads must be finite. Two properties make it safe — it
        touches only cells (``depth <= 0``) whose value no *bulk* cell ever
        reads, and it is idempotent, so running it twice leaves the field
        bitwise where the first one left it.

        Unconditional: the production caller is :meth:`ensure_pinned`, which
        runs it once per ``(field, method, lev, grid_version)`` — at
        classification, not per evaluate (B25).

        Expressed as ``nnz = 0`` rows through the band kernel
        (:func:`~blockamr.ibm.band_rows.pin_rows`), so it writes device memory
        without a second kernel and without staging the field through the host.
        """
        import blockamr

        table = self._pin_table(lev, field.ncomp)
        if table is None:
            return
        blockamr.apply_band_rows(
            field.mf[lev],
            field.mf[lev],
            table,
            field.ncomp,
            blockamr.BandMode.Overwrite,
            1.0,
            self.grid_version,
        )

    def invalidate(self):
        """Drop everything and start a new generation.

        Called by the mesh on a regrid and on a ``bodies`` re-assignment; call
        it directly after changing the geometry any other way.
        """
        self._mesh._invalidate_ibm()

    def _pin_table(self, lev, ncomp):
        """The pin table of ``(lev, ncomp)``, or ``None`` when nothing is solid."""
        key = (lev, int(ncomp), self.grid_version)
        if key not in self._pin_tables:
            rows = pin_rows(self._boxes(lev), self.geometry(lev), int(ncomp))
            self._pin_tables[key] = None if rows.nrows == 0 else band_table(rows, self.grid_version)
        return self._pin_tables[key]

    def _boxes(self, lev):
        """The level's local boxes, in ``MFIterator`` order."""
        key = (lev, self.grid_version)
        if key not in self._grids:
            self._grids[key] = box_grids(self._mesh, lev)
        return self._grids[key]

    def _clear(self):
        """Drop the cached objects (the mesh has already bumped the version)."""
        self._grids.clear()
        self._geometry.clear()
        self._bands.clear()
        self._method_data.clear()
        self._pin_tables.clear()
        self._pinned.clear()
