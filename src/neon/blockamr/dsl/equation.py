# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Implicit equation: lhs == rhs.

Created by imp.laplacian(sigma, p) == exp.div(U).
Dispatched to MLMG by solve().
"""


class Equation:
    """Implicit equation for solve() dispatch.

    lhs: ImplicitLaplacian (or future implicit operators)
    rhs: CellDivergence (or future explicit RHS operators)
    """

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
