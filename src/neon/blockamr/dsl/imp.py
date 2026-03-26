# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Implicit DSL operators (cf. OpenFOAM fvm::).

These create lazy operator objects that are resolved to AMReX MLMG solves
when passed to solve().
"""


class ImplicitLaplacian:
    """Implicit Laplacian: div(sigma * grad(field)).

    Created by imp.laplacian(sigma, field). The actual MLNodeLaplacian + MLMG
    are set up lazily on first solve.
    """

    _name = "ImplicitLaplacian"

    def __init__(self, sigma, field):
        self.sigma = sigma
        self.field = field

    def __eq__(self, rhs):
        from .equation import Equation
        return Equation(lhs=self, rhs=rhs)


def laplacian(sigma, field):
    """imp.laplacian(sigma, p) — implicit Laplacian for pressure solve."""
    return ImplicitLaplacian(sigma, field)
