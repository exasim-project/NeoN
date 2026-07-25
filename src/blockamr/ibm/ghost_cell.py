# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``ghostCell`` — a sharp-interface method, expressed entirely as wall rows.

The whole method is the geometry in :mod:`blockamr.ibm.rows` plus the generic
``P``/``R`` kernels: there is no ``ghostCell`` kernel. Ghost cells get a
reconstruction row built from the image point; deeper solid cells get a
``b = 0`` row that pins them to zero, and ``R`` zeroes the operator result over
the same target set (the operator's value there is meaningless).
"""

import blockamr

from .rows import ghost_cell_rows


class GhostCell:
    """Operator method: reconstruct wall values into a work buffer, then run the
    plain operator against it."""

    kind = "operator"
    restrict_mode = "Zero"
    requires_bodies = True

    @staticmethod
    def build_tables(mesh, field):
        """One :class:`blockamr.WallTable` per level, cached on the field and
        rebuilt when the grid generation changes.

        Keyed per (method, field): ``w``/``b`` are geometry and would be shared,
        but ``gamma`` is the field's own ``ibm_bc`` datum — so a cache keyed by
        mesh alone passes every single-field test and fails the two-field one.
        """
        cache = field._ibm_cache
        entry = cache.get("ghostCell")
        if entry is None or entry[0] != mesh.grid_version:
            tables = {
                lev: blockamr.WallTable(
                    **ghost_cell_rows(mesh, lev, mesh.bodies, field.ibm_bc, field.ncomp),
                    grid_version=mesh.grid_version,
                )
                for lev in range(mesh.n_levels())
            }
            entry = (mesh.grid_version, tables)
            cache["ghostCell"] = entry
        return entry[1]
