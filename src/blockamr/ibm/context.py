# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The evaluate-time IBM driver: ``P`` before the operator, ``R`` after it.

One schedule serves every method (see ``plans/IBM/ibm-row-format.md`` §6)::

    FillBoundary(phi) -> copy to work -> P(work) -> FillBoundary(work)
                      -> bulk sweep (the untouched operator kernel) -> R(result)

Both ``FillBoundary``s are load-bearing: donors live in neighbours' halos, and
``P`` changes valid cells that neighbours see.

The operators never learn that IBM exists — they are handed a *different field*
(:class:`WorkFieldView`), not a modified kernel. That substitution is the whole
of the design's "adding a method costs zero C++" claim.
"""

import blockamr


class WorkFieldView:
    """A ``CellField`` stand-in whose ``mf`` is the IBM work MultiFab.

    Kernels read the field they are handed (``cell_field.mf[lev]``), never
    ``sp_op.field``, so substituting this view is enough to make every existing
    operator evaluate against the wall-reconstructed field.
    """

    def __init__(self, base, mf_list):
        self._base = base
        self.mf = mf_list

    def __getattr__(self, name):
        if name == "_base":  # not yet bound — do not recurse
            raise AttributeError(name)
        return getattr(self._base, name)


class IbmEvaluation:
    """Per-``evaluate`` IBM state for one field: the tables, the work buffers,
    and the two kernel calls that bracket the operator sweep.

    ``method is None`` (no ``solution["ibm"]``) and ``noIbm`` both yield
    :meth:`prolong` returning the field itself and :meth:`restrict` doing
    nothing, so the opt-out path is bitwise identical to the plain operator by
    *construction* rather than by care.
    """

    def __init__(self, method, name, cell_field):
        self.method = method
        self.name = name
        self.field = cell_field
        self.mesh = cell_field.mesh
        self.tables = None if method is None else method.build_tables(self.mesh, cell_field)
        self.work = None if self.tables is None else WorkFieldView(cell_field, self._work_mfs())

    def prolong(self, lev):
        """``P``: the wall-reconstructed field the operator should read."""
        if self.tables is None:
            return self.field
        work_mf = self.work.mf[lev]
        blockamr.copy_multifab(work_mf, self.field.mf[lev], self.field.ncomp, self.field.ngrow)
        blockamr.apply_wall_stencils(
            work_mf, self.tables[lev], self.field.ncomp, 1.0, self.mesh.grid_version
        )
        work_mf.fill_boundary(self.mesh.geom(lev))
        return self.work

    def restrict(self, out_mf, lev):
        """``R``: what happens to the operator result in non-fluid cells."""
        if self.tables is None:
            return
        blockamr.restrict_band(
            out_mf,
            self.tables[lev],
            _restrict_mode(self.method.restrict_mode),
            self.field.ncomp,
            1.0,
            self.mesh.grid_version,
        )

    # -- internals ----------------------------------------------------------

    def _work_mfs(self):
        """Work buffers matching the field's layout, cached on the field and
        rebuilt whenever the grid generation changes."""
        cache = self.field._ibm_cache
        entry = cache.get("work")
        if entry is None or entry[0] != self.mesh.grid_version:
            mfs = [
                blockamr.MultiFab(
                    self.mesh.box_array(lev),
                    self.mesh.dm(lev),
                    self.field.ncomp,
                    self.field.ngrow,
                    memory=self.field._memory,
                )
                for lev in range(self.mesh.n_levels())
            ]
            entry = (self.mesh.grid_version, mfs)
            cache["work"] = entry
        return entry[1]


def _restrict_mode(name):
    """Resolve a method's ``restrict_mode`` string to the C++ enum.

    Looked up on call, not at import: ``blockamr.RestrictMode`` is registered by
    the extension module, which is still initialising when this package loads.
    """
    modes = {
        "Zero": blockamr.RestrictMode.Zero,
        "Overwrite": blockamr.RestrictMode.Overwrite,
        "AddSource": blockamr.RestrictMode.AddSource,
    }
    if name not in modes:
        raise ValueError(f"Unknown restrict mode {name!r}; valid modes: {sorted(modes)}")
    return modes[name]
