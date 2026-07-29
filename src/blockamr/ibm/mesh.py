# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``mesh.ibm`` — the shared interface between preprocessing and the schemes (B2).

Everything the IBM precomputes from ``mesh.bodies`` hangs here: the per-cell
:class:`~blockamr.ibm.geometry.IbmGeometry`, the packed geometry fab, the
``SOLID | WALL | FLUID`` marker and each method's own data. Nothing is built
until it is asked for, and every cache is keyed by the grid generation
(design §8), so a regrid or a moved body cannot be served from a stale entry::

    mesh.bodies = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}
    ct = mesh.ibm.cell_type(GhostCell, lev)     # classifies on first use
    donors = mesh.ibm.data(GhostCell, lev)      # the method's own extras

The table design §8 fixes is the whole of it, and the row that is **missing** is
the point: there is no band, no width, no stencil shape and no row table here
any more.

Method data is stored **opaquely**: the method declares the type, the mesh
allocates and invalidates it and never looks inside (design §2.4).
"""

import weakref

from .classify import _patches, box_grids
from .geometry import GEOM_NCOMP, geometry_on_grids, packed_geometry_on_grids


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
        self._geometry_fabs = {}
        self._cell_types = {}
        self._method_data = {}
        self._wall_data = {}
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

    def geometry_fab(self, lev, ngrow):
        """The v2 packed geometry ``MultiFab`` of ``lev`` (B29, design §2.1).

        The compiled side's ``IbmGeometryFab``: one ``MultiFab`` of
        :data:`~blockamr.ibm.geometry.GEOM_NCOMP` components, filled over the
        **grown** box so that ``blockamr.classify_default`` can classify a grown
        box in a single pass (review F10 — see
        :func:`~blockamr.ibm.geometry.packed_box_geometry` for the two halves of
        the ghost contract this honours).

        Separate from :meth:`geometry`, which keeps returning v1's per-box
        :class:`~blockamr.ibm.geometry.IbmGeometry` dataclasses — ``depth`` and
        all — until B36/B37 rewire their readers. Nothing on a v1 evaluate path
        calls this yet.

        Cached on ``(lev, grid_version)``, i.e. design §8's key with **no
        ``ngrow``** in it, and grown **monotonically**: a wider request rebuilds
        the fab, a narrower one is served the wider fab it already has. That is
        what lets the documented cache key stand while callers with different
        stencil reaches share one geometry — a fab is never shrunk under a
        caller that asked for more.
        """
        import blockamr

        ngrow = int(ngrow)
        key = (lev, self.grid_version)
        have = self._geometry_fabs.get(key)
        if have is None or have[0] < ngrow:
            grids = self._boxes(lev)
            blocks = packed_geometry_on_grids(grids, self.bodies, ngrow)
            mf = blockamr.MultiFab(self._mesh.box_array(lev), self._mesh.dm(lev), GEOM_NCOMP, ngrow)
            for mfi, block in zip(blockamr.MFIterator(mf), blocks):
                mf.copy_grown_from(mfi, block)
            self._geometry_fabs[key] = (ngrow, mf)
        return self._geometry_fabs[key][1]

    def cell_type(self, method, lev, ngrow=1):
        """The v2 marker of ``lev`` under ``method`` (design §2.2, §8).

        ``SOLID | WALL | FLUID`` on this level's grids, filled by the method's
        own classification — ``blockamr.classify_default`` unless the method
        declares a ``classify``, which is conformance check M4.

        Cached on ``(method, lev, grid_version)`` — design §8's key, with **no
        ``ngrow``** in it — and grown monotonically, exactly as
        :meth:`geometry_fab` is and for the same reason: W1's siblings read the
        marker at their own stencil reach, so two equations on one level can
        want different ghost widths of one marker. ``MARKER_NGROW`` is the
        default classification's floor, not an allocation size.
        """
        import blockamr

        ngrow = int(ngrow)
        key = (method, int(lev), self.grid_version)
        have = self._cell_types.get(key)
        if have is None or have[0] < ngrow:
            # The two geometry validity checks live on the v1 classification
            # (`classify.py`'s `_check_adjacent` and `_check_resolvable_gap`)
            # and the packed fab does not repeat them, by its own note. They are
            # api §9's "two bodies less than a cell apart" and "the body is
            # incompatible with this mesh", and §9 places both at
            # *classification* — which is here. Until they are compiled, this is
            # what runs them: once per (lev, generation), cached, and on exactly
            # the path that used to reach them through `ibm.band(...)`.
            self.geometry(lev)
            geom = self._mesh.geom(lev)
            ct = blockamr.CellTypeFab(self._mesh.box_array(lev), self._mesh.dm(lev), ngrow)
            classify = getattr(method, "classify", None) or blockamr.classify_default
            classify(ct, self.geometry_fab(lev, ngrow), geom)
            self._cell_types[key] = (ngrow, ct)
        return self._cell_types[key][1]

    def wall_data(self, method, lev, ngrow=1):
        """The method's own **device-side** data, built once per generation.

        The v2 peer of :meth:`data`, and stored just as opaquely: the method
        declares what it precomputes from the marker and the geometry, the mesh
        allocates and invalidates it and never looks inside (design §2.3).
        """
        key = (method, int(lev), self.grid_version)
        if key not in self._wall_data:
            names, _bodies = _patches(self.bodies)
            self._wall_data[key] = method.wall_preprocess(
                self.cell_type(method, lev, ngrow),
                self.geometry_fab(lev, ngrow),
                self._mesh.geom(lev),
                names,
            )
        return self._wall_data[key]

    def data(self, method, lev):
        """The method's own preprocessed data, built once per generation.

        The result is whatever the method's ``preprocess`` returned and is
        stored as-is; nothing here reads a field of it.
        """
        key = (method, lev, self.grid_version)
        if key not in self._method_data:
            self._method_data[key] = method.preprocess(self._mesh, lev)
        return self._method_data[key]

    def ensure_pinned(self, field, method, lev, ngrow=1):
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
        self.pin_non_fluid(field, lev, method, ngrow)
        seen.add(key)

    def pin_non_fluid(self, field, lev, method, ngrow=1):
        """Write the pin value into every ``SOLID`` cell of ``field`` (B7).

        The one write this design makes to a user field, and design §7 is where
        it is argued: the interior sweep reads solid neighbours at a wall cell,
        and those reads must be finite. Two properties make it safe — it
        touches only cells whose value no *fluid* cell ever reads, and it is
        idempotent, so running it twice leaves the field bitwise where the first
        one left it.

        Unconditional: the production caller is :meth:`ensure_pinned`, which
        runs it once per ``(field, method, lev, grid_version)`` — at
        classification, not per evaluate (B25).

        It is design §7's four compiled lines, ``blockamr.pin_solid``, on the
        method's own marker. v1 expressed the same write as a table of
        ``nnz = 0`` rows so it could reuse the band kernel; the marker makes it
        a kernel that reads nothing but one ``uint8`` per cell.
        """
        import blockamr

        blockamr.pin_solid(field.mf[lev], self.cell_type(method, lev, ngrow), 0.0, field.ncomp)

    def invalidate(self):
        """Drop everything and start a new generation.

        Called by the mesh on a regrid and on a ``bodies`` re-assignment; call
        it directly after changing the geometry any other way.
        """
        self._mesh._invalidate_ibm()

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
        self._geometry_fabs.clear()
        self._cell_types.clear()
        self._method_data.clear()
        self._wall_data.clear()
        self._pinned.clear()
