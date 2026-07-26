# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Implicit DSL operators (cf. OpenFOAM fvm::).

These create lazy operator objects that are resolved to AMReX MLMG solves
when passed to solve().
"""

from .eqterm import EqTerm


class ImplicitLaplacian(EqTerm):
    """Implicit Laplacian: div(sigma * grad(field)).

    Created by imp.laplacian(sigma, field). The actual MLNodeLaplacian + MLMG
    are set up lazily on first solve. ``== rhs`` builds an implicit Equation
    (via EqTerm.__eq__).
    """

    kind = "implicit"
    _scheme_operator = "laplacian"
    scheme_key = "laplacian"

    def __init__(self, sigma, field):
        super().__init__(field, coefficient=sigma)
        self.sigma = sigma


def laplacian(sigma, field):
    """imp.laplacian(sigma, p) — implicit Laplacian for pressure solve."""
    return ImplicitLaplacian(sigma, field)
