# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Explicit DSL operators (cf. OpenFOAM fvc::).

These create operator objects that build JAX kernels for stencil computation.
"""

from ..operators.ddt import Ddt
from ..operators.div import Div
from ..operators.grad import Grad
from ..operators.laplacian import Laplacian
from ..operators.source import Source
from .eqterm import EqTerm


def ddt(field):
    return Ddt(field)


def div(face_fluxes_or_field, field=None, scheme=None):
    """Divergence operator.

    Two forms:
      exp.div(phi, U)  — advective flux divergence (existing Div operator)
      exp.div(U)       — cell velocity divergence for pressure RHS
    """
    if field is None:
        return CellDivergence(face_fluxes_or_field)
    else:
        return Div(face_fluxes_or_field, field, scheme=scheme)


def grad(field, scheme=None):
    """Gradient operator.

    For a pressure NodalField after an implicit solve, returns a
    PressureGradient that reads the stored gradient (from getFluxes).
    Otherwise returns the standard explicit Grad operator.
    """
    if hasattr(field, "grad") and field.grad is not None:
        return PressureGradient(field)
    return Grad(field, scheme=scheme)


def laplacian(gamma_func, field, scheme=None):
    return Laplacian(gamma_func, field, scheme=scheme)


def source(coeff_func, field):
    return Source(coeff_func, field)


# ---------------------------------------------------------------------------
# Cell velocity divergence (for pressure RHS)
# ---------------------------------------------------------------------------


class CellDivergence(EqTerm):
    """Divergence of a cell-centred velocity field (ncomp=3).

    Used as RHS of pressure equation: imp.laplacian(sigma, p) == exp.div(U).
    Evaluated inside solve() via MLNodeLaplacian.compDivergence.
    """

    kind = "spatial"

    def __init__(self, vel_field):
        super().__init__(vel_field)
        self.vel_field = vel_field

    @property
    def scheme_key(self):
        return f"div({self._named(self.vel_field, 'velocity')})"


# ---------------------------------------------------------------------------
# Pressure gradient (lazy, reads stored result from implicit solve)
# ---------------------------------------------------------------------------


class PressureGradient:
    """Lazy reference to the pressure gradient stored after an implicit solve.

    exp.grad(p) returns this when p has a stored gradient.
    Supports -dt * exp.grad(p) via __rmul__ and __neg__.
    p_field.grad[lev] is a list of per-box JAX arrays.
    """

    def __init__(self, p_field):
        self.p_field = p_field

    def __rmul__(self, scalar):
        return ScaledPressureGradient(scalar, self.p_field)

    def __neg__(self):
        return ScaledPressureGradient(-1.0, self.p_field)

    def evaluate(self, lev=0):
        return self.p_field.grad[lev]


class ScaledPressureGradient:
    """Scaled pressure gradient: scalar * grad(p)."""

    def __init__(self, scalar, p_field):
        self.scalar = scalar
        self.p_field = p_field

    def __rmul__(self, scalar):
        return ScaledPressureGradient(self.scalar * scalar, self.p_field)

    def __neg__(self):
        return ScaledPressureGradient(-self.scalar, self.p_field)

    def evaluate(self, lev=0):
        return [self.scalar * g for g in self.p_field.grad[lev]]
