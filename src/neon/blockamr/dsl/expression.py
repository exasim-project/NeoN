# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Deprecated back-compat shim: Expression is now the unified Equation.

Kept as a thin adapter so old call sites keep working; deleted in plan 06.
"""

from .equation import Equation


class Expression(Equation):
    """Back-compat alias for the unified Equation. Use Equation directly."""
