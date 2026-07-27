# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr

_SINGLE_BODY_KEY = "body"


def _body_property():
    """The pre-multi-patch ``mesh.body`` singular alias over ``mesh.bodies``.

    Immersed geometry is patch-keyed (``mesh.bodies``) so more than one body is
    expressible; ``mesh.body = c`` remains as the single-body shorthand and
    stores it under the reserved patch name ``"body"``.
    """

    def _get(self):
        if len(self.bodies) != 1:
            return None
        return next(iter(self.bodies.values()))

    def _set(self, value):
        self.bodies = {} if value is None else {_SINGLE_BODY_KEY: value}

    return property(_get, _set)


class _ImmersedGeometry:
    """The mesh's immersed-body surface: ``bodies``, ``body`` and ``ibm``.

    ``bodies`` is the patch-keyed geometry (API doc §6), and everything derived
    from it lives behind the lazy ``mesh.ibm`` (``plans/IBM/design.md`` §8):
    nothing is classified until a scheme asks, and re-assigning ``bodies``
    starts a new IBM generation so no cache can serve a moved body.

    The IBM generation is deliberately *not* ``grid_version``: that one is the
    box layout's version, which the compiled kernels compare against, and a
    body that moves on an unchanged layout must not look like a regrid to them.
    """

    def _init_immersed(self):
        self._bodies = {}
        self._ibm = None
        self._ibm_generation = 0

    body = _body_property()

    @property
    def bodies(self):
        """Immersed bodies, keyed by patch name; re-assign to move them."""
        return self._bodies

    @bodies.setter
    def bodies(self, value):
        self._bodies = value
        self._invalidate_ibm()

    @property
    def ibm(self):
        """The lazy IBM cache of this mesh (:class:`~blockamr.ibm.mesh.IbmMesh`)."""
        if self._ibm is None:
            # deferred: blockamr.ibm pulls in jax and the methods, and mesh.py
            # is imported while the package is still assembling.
            from .ibm.mesh import IbmMesh

            self._ibm = IbmMesh(self)
        return self._ibm

    def _invalidate_ibm(self):
        """Start a new IBM generation and drop what the old one built."""
        self._ibm_generation += 1
        if self._ibm is not None:
            self._ibm._clear()


class Mesh(_ImmersedGeometry):
    """Single-level mesh. Same interface as AmrMesh."""

    def __init__(self, ba, dm, geom):
        self._ba = ba
        self._dm = dm
        self._geom = geom
        self._fields = []
        # Immersed-body geometry + precomputed per-method IBM data (API doc
        # §6). ``bodies`` is a patch-keyed dict set by the caller (e.g. the
        # mesh factory, from meshDict); ``build_ibm``/``ibm_data`` below.
        self._init_immersed()
        self._ibm_data = {}
        # Bumped on every regrid; a WallTable built from an older generation is
        # rejected by the kernels rather than computing plausible wrong numbers
        # (see plans/IBM/ibm-row-format.md §3). A single-level Mesh never
        # regrids, so this stays 0.
        self.grid_version = 0

    @property
    def max_level(self):
        return 0

    def n_levels(self):
        return 1

    def finest_level(self):
        return 0

    def geom(self, lev):
        return self._geom

    def box_array(self, lev):
        return self._ba

    def dm(self, lev):
        return self._dm

    def register_field(self, field):
        self._fields.append(field)
        field._on_new_level(0, self._ba, self._dm)

    # ------------------------------------------------------------------
    # Immersed body (API doc §6): geometry on ``self.body``, per-method
    # data precomputed eagerly by ``build_ibm`` and read back by
    # ``ibm_data``. Single-level ``Mesh`` never regrids, so there is no
    # rebuild hook here (see ``AmrMesh.regrid``).
    # ------------------------------------------------------------------

    def build_ibm(self, methods):
        """Eagerly precompute each method's data (masks/fractions) from
        ``self.body``. ``methods`` is a list of IBM strategy classes (e.g.
        ``[DirectForcing]``, or via ``IBM.lookup(name)``)."""
        if self.body is None:
            raise ValueError(
                "mesh.body must be set (or mesh.bodies must hold exactly one "
                "body) before build_ibm(...)"
            )
        self._ibm_methods = list(methods)
        self._ibm_data = {method: method.build_data(self, self.body) for method in methods}

    def ibm_data(self, method):
        """Return the precomputed data for ``method`` (as built by
        ``build_ibm``); raises a clear error when it hasn't been built."""
        data = self._ibm_data.get(method)
        if data is None:
            name = getattr(method, "__name__", method)
            raise RuntimeError(
                f"IBM data for '{name}' not built; call mesh.build_ibm([...]) first."
            )
        return data


class _AmrCoreDelegate(blockamr.AmrCore):
    """Forwards AmrCore virtuals to owning AmrMesh."""

    def __init__(self, geom, amr_info, owner):
        super().__init__(geom, amr_info)
        self._owner = owner

    def make_new_level_from_scratch(self, lev, time, ba, dm):
        self._owner._on_new_level(lev, time, ba, dm)

    def make_new_level_from_coarse(self, lev, time, ba, dm):
        self._owner._on_new_level_from_coarse(lev, time, ba, dm)

    def remake_level(self, lev, time, ba, dm):
        self._owner._on_remake_level(lev, time, ba, dm)

    def clear_level(self, lev):
        self._owner._on_clear_level(lev)

    def error_est(self, lev, tags, time, ngrow):
        self._owner._on_error_est(lev, tags, time, ngrow)


class AmrMesh(_ImmersedGeometry):
    """High-level AMR mesh managing fields and their lifecycle callbacks."""

    def __init__(self, geom, amr_info):
        self._core = _AmrCoreDelegate(geom, amr_info, owner=self)
        self._fields = []
        self._tag_func = None
        # Immersed-body geometry + precomputed per-method IBM data (API doc
        # §6). ``bodies`` is a patch-keyed dict set by the caller (e.g. the
        # mesh factory, from meshDict); ``build_ibm``/``ibm_data`` below.
        self._init_immersed()
        self._ibm_data = {}
        # Bumped on every regrid — see the note on ``Mesh.grid_version``.
        self.grid_version = 0

    # Metadata delegates
    def n_levels(self):
        return self._core.finest_level + 1

    def finest_level(self):
        return self._core.finest_level

    @property
    def max_level(self):
        return self._core.max_level

    def geom(self, lev):
        return self._core.geom(lev)

    def box_array(self, lev):
        return self._core.box_array(lev)

    def dm(self, lev):
        return self._core.dm(lev)

    def ref_ratio(self, lev):
        return self._core.ref_ratio(lev)

    # Field registration
    def register_field(self, field):
        self._fields.append(field)

    # Lifecycle
    def init_from_scratch(self, time):
        self._core.init_from_scratch(time)

    def regrid(self, t, tag):
        self._tag_func = tag
        self._core.regrid(0, t)
        self.grid_version += 1
        self._invalidate_ibm()
        self._rebuild_ibm()

    # ------------------------------------------------------------------
    # Immersed body (API doc §6): geometry on ``self.body``, per-method
    # data precomputed eagerly by ``build_ibm``, read back by ``ibm_data``,
    # and rebuilt for the new box arrays after every regrid.
    # ------------------------------------------------------------------

    def build_ibm(self, methods):
        """Eagerly precompute each method's data (masks/fractions) from
        ``self.body``. ``methods`` is a list of IBM strategy classes (e.g.
        ``[DirectForcing]``, or via ``IBM.lookup(name)``)."""
        if self.body is None:
            raise ValueError(
                "mesh.body must be set (or mesh.bodies must hold exactly one "
                "body) before build_ibm(...)"
            )
        self._ibm_methods = list(methods)
        self._ibm_data = {method: method.build_data(self, self.body) for method in methods}

    def ibm_data(self, method):
        """Return the precomputed data for ``method`` (as built by
        ``build_ibm``); raises a clear error when it hasn't been built."""
        data = self._ibm_data.get(method)
        if data is None:
            name = getattr(method, "__name__", method)
            raise RuntimeError(
                f"IBM data for '{name}' not built; call mesh.build_ibm([...]) first."
            )
        return data

    def _rebuild_ibm(self):
        """Recompute per-method IBM data for the current box arrays (regrid
        hook). Masks are spatial and must be rebuilt; ``force_history`` is a
        time series and is carried forward onto the rebuilt data."""
        if not getattr(self, "_ibm_methods", None):
            return
        for method in self._ibm_methods:
            old = self._ibm_data[method]
            new = method.build_data(self, self.body)
            new.force_history = old.force_history
            self._ibm_data[method] = new

    # Callbacks — dispatch to fields
    def _on_new_level(self, lev, time, ba, dm):
        for f in self._fields:
            f._on_new_level(lev, ba, dm)

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        for f in self._fields:
            f._on_new_level_from_coarse(lev, time, ba, dm)

    def _on_remake_level(self, lev, time, ba, dm):
        for f in self._fields:
            f._on_remake_level(lev, time, ba, dm)

    def _on_clear_level(self, lev):
        for f in self._fields:
            f._on_clear_level(lev)

    def _on_error_est(self, lev, tags, time, ngrow):
        if self._tag_func:
            self._tag_func(lev, tags, time, ngrow)

    # Plotfile
    def write_plotfile(self, name, phi, time):
        nlevels = self.finest_level() + 1
        blockamr.write_multilevel_plotfile(
            name,
            nlevels,
            [phi.mf[lev] for lev in range(nlevels)],
            [phi.name],
            [self.geom(lev) for lev in range(nlevels)],
            time,
            [0] * nlevels,
            [self.ref_ratio(lev) for lev in range(nlevels - 1)],
        )
