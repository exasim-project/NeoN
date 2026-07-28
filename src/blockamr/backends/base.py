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
        """In-place forward-Euler update of ``cell_field`` on level ``lev``.

        The no-IBM step only, call-for-call the plain operator's (the
        "absent ⇒ bitwise the plain operator" contract, api §1). An active
        band never lands here: ``solve()`` routes it through the driver's
        ``source_level`` (which owns the pin) plus ``blockamr.euler_update``
        — never a fused step kernel (row-format rule R4).
        """
        ...

    def source(self, terms, cell_field, lev, t, ibm=None):
        """The accumulated source MultiFab ``Σ coeff·op(phi)`` on level ``lev``.

        The R4 seam made explicit: operator evaluation and time update are
        separate named launches, so a wall sweep (``ibm.apply``) fits between
        them and RK stages can consume the source without an update. The
        returned MultiFab is on the level's box array; it may be a scratch
        buffer reused by the next ``source`` call, so consume it first.
        """
        ...

    def evaluate(self, terms, cell_field, lev, t, ibm=None) -> list:
        """Per-box source arrays for the spatial ``terms`` on level ``lev``.

        ``ibm``, when given, is the wall driver
        (:class:`~blockamr.ibm.driver.WallEvaluation`): once every term's
        interior sweep has run, ``ibm.apply(result_mf, lev, t)`` writes the wall
        cells through the boundary schemes and masks the solid. ``None`` — the
        no-``"ibm"`` path, ``noIbm`` and an empty band — is one branch outside
        the kernel, which is what keeps those results bitwise the plain
        operator's.
        """
        ...
