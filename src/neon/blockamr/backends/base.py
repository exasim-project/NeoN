# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import Protocol


class ExplicitBackend(Protocol):
    """Explicit forward-Euler discretisation backend for one AMR level.

    The caller (``dsl.solve.solve``/``dsl.solve.evaluate``) owns the level
    loop, ``fill_patch``, and ``average_down`` — a backend only touches a
    single level, so ghost/BC handling stays backend-agnostic.
    """

    def euler_step(self, equation, cell_field, lev, t, dt) -> None:
        """In-place forward-Euler update of ``cell_field`` on level ``lev``."""
        ...

    def evaluate(self, terms, cell_field, lev, t) -> list:
        """Per-box source arrays for the spatial ``terms`` on level ``lev``."""
        ...
