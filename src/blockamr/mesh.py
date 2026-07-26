# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr


class Mesh:
    """Single-level mesh. Same interface as AmrMesh."""

    def __init__(self, ba, dm, geom):
        self._ba = ba
        self._dm = dm
        self._geom = geom
        self._fields = []
        # Immersed-body geometry + precomputed per-method IBM data (API doc
        # §6). ``body`` is set by the caller (e.g. the mesh factory, from
        # meshDict); ``build_ibm``/``ibm_data`` below.
        self.body = None
        self._ibm_data = {}

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
            raise ValueError("mesh.body must be set before build_ibm(...)")
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


class AmrMesh:
    """High-level AMR mesh managing fields and their lifecycle callbacks."""

    def __init__(self, geom, amr_info):
        self._core = _AmrCoreDelegate(geom, amr_info, owner=self)
        self._fields = []
        self._tag_func = None
        # Immersed-body geometry + precomputed per-method IBM data (API doc
        # §6). ``body`` is set by the caller (e.g. the mesh factory, from
        # meshDict); ``build_ibm``/``ibm_data`` below.
        self.body = None
        self._ibm_data = {}

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
            raise ValueError("mesh.body must be set before build_ibm(...)")
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
