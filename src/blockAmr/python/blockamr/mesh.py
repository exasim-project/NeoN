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


