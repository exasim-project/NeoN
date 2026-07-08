# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon.blockamr as blockamr


class Mesh:
    """Single-level mesh. Same interface as AmrMesh.

    Optional ``eb_factory`` enables embedded-boundary support. When set,
    ``has_eb`` is True and downstream code (CellField, dsl_solver) selects
    EB-aware MultiFab allocation and EB linear operators.
    """

    def __init__(self, ba, dm, geom, eb_factory=None):
        self._ba = ba
        self._dm = dm
        self._geom = geom
        self._eb_factory = eb_factory
        self._fields = []

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

    @property
    def has_eb(self):
        return self._eb_factory is not None

    def eb_factory(self, lev=0):
        """Return the EBFArrayBoxFactory for ``lev`` (or None when no EB)."""
        return self._eb_factory

    def vol_frac(self, lev=0):
        """Return per-box jnp arrays of valid-cell volume fractions for ``lev``.

        For EB meshes the values come from EBFArrayBoxFactory.vol_frac() and
        live in [0, 1] (covered=0, regular=1, cut cells in between). For
        non-EB meshes the helper returns ``None`` so callers can elide the
        volfrac multiply at zero overhead.
        """
        if not self.has_eb:
            return None
        if not hasattr(self, '_vol_frac_cache'):
            self._vol_frac_cache = {}
        cached = self._vol_frac_cache.get(lev)
        if cached is not None:
            return cached
        import jax.numpy as jnp
        import numpy as np
        vf_mf = self._eb_factory.vol_frac()
        vf_ng = vf_mf.n_grow()
        result = []
        for arr, m in zip(vf_mf.arrays(), vf_mf.fab_metadata()):
            Nx, Ny, Nz = m[1], m[2], m[3]
            vNx, vNy, vNz = Nx - 2 * vf_ng, Ny - 2 * vf_ng, Nz - 2 * vf_ng
            valid = np.asarray(arr)[
                vf_ng:vf_ng + vNx,
                vf_ng:vf_ng + vNy,
                vf_ng:vf_ng + vNz,
                0,
            ]
            result.append(jnp.asarray(valid))
        self._vol_frac_cache[lev] = result
        return result

    def register_field(self, field):
        self._fields.append(field)
        field._on_new_level(0, self._ba, self._dm)


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
    """High-level AMR mesh managing fields and their lifecycle callbacks.

    Optional ``eb_factory_factory`` is a callable
    ``(lev, geom, ba, dm) -> EBFArrayBoxFactory`` invoked on each new level
    to build a per-level EB factory. Pass ``None`` for non-EB meshes.
    """

    def __init__(self, geom, amr_info, eb_factory_factory=None):
        self._core = _AmrCoreDelegate(geom, amr_info, owner=self)
        self._fields = []
        self._tag_func = None
        self._eb_factory_factory = eb_factory_factory
        self._eb_factories = {}

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

    @property
    def has_eb(self):
        return self._eb_factory_factory is not None

    def eb_factory(self, lev):
        return self._eb_factories.get(lev)

    def vol_frac(self, lev):
        if not self.has_eb:
            return None
        if not hasattr(self, '_vol_frac_cache'):
            self._vol_frac_cache = {}
        cached = self._vol_frac_cache.get(lev)
        if cached is not None:
            return cached
        import jax.numpy as jnp
        import numpy as np
        ebf = self._eb_factories.get(lev)
        if ebf is None:
            return None
        vf_mf = ebf.vol_frac()
        vf_ng = vf_mf.n_grow()
        result = []
        for arr, m in zip(vf_mf.arrays(), vf_mf.fab_metadata()):
            Nx, Ny, Nz = m[1], m[2], m[3]
            vNx, vNy, vNz = Nx - 2 * vf_ng, Ny - 2 * vf_ng, Nz - 2 * vf_ng
            valid = np.asarray(arr)[
                vf_ng:vf_ng + vNx,
                vf_ng:vf_ng + vNy,
                vf_ng:vf_ng + vNz,
                0,
            ]
            result.append(jnp.asarray(valid))
        self._vol_frac_cache[lev] = result
        return result

    # Field registration
    def register_field(self, field):
        self._fields.append(field)

    # Lifecycle
    def init_from_scratch(self, time):
        self._core.init_from_scratch(time)

    def regrid(self, t, tag):
        self._tag_func = tag
        self._core.regrid(0, t)

    # Callbacks — dispatch to fields
    def _build_eb_factory(self, lev, ba, dm):
        if self._eb_factory_factory is not None:
            self._eb_factories[lev] = self._eb_factory_factory(
                lev, self.geom(lev), ba, dm)

    def _on_new_level(self, lev, time, ba, dm):
        self._build_eb_factory(lev, ba, dm)
        for f in self._fields:
            f._on_new_level(lev, ba, dm)

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        self._build_eb_factory(lev, ba, dm)
        for f in self._fields:
            f._on_new_level_from_coarse(lev, time, ba, dm)

    def _on_remake_level(self, lev, time, ba, dm):
        self._build_eb_factory(lev, ba, dm)
        for f in self._fields:
            f._on_remake_level(lev, time, ba, dm)

    def _on_clear_level(self, lev):
        self._eb_factories.pop(lev, None)
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
