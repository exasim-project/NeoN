# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Explicit DSL operators (cf. OpenFOAM fvc::)."""

from ..operators.ddt import Ddt
from ..operators.div import Div
from ..operators.grad import Grad
from ..operators.laplacian import Laplacian
from ..operators.source import Source
from .eqterm import EqTerm


def ddt(field):
    return Ddt(field)


def div(face_fluxes_or_field, field=None):
    """Divergence: ``exp.div(phi, U)`` advective flux, or ``exp.div(U)`` cell velocity.

    The scheme is resolved by name from the equation's ``schemes`` at solve time;
    construct ``Div`` directly to pin a scheme object instead.
    """
    if field is None:
        return CellDivergence(face_fluxes_or_field)
    else:
        return Div(face_fluxes_or_field, field)


def grad(field):
    """Gradient. A field carrying a stored gradient (post-implicit-solve pressure)
    yields a :class:`PressureGradient` reading it; anything else yields ``Grad``.
    """
    if hasattr(field, "grad") and field.grad is not None:
        return PressureGradient(field)
    return Grad(field)


def laplacian(gamma_func, field):
    return Laplacian(gamma_func, field)


def source(coeff_func, field):
    return Source(coeff_func, field)


class CellDivergence(EqTerm):
    """Divergence of a CELL-CENTRED velocity field (ncomp=3).

    The RHS of ``imp.laplacian(sigma, p) == exp.div(U)``; evaluated inside ``solve()``
    by ``MLNodeLaplacian.compDivergence``, which makes it NODAL.
    """

    kind = "spatial"

    def __init__(self, vel_field):
        super().__init__(vel_field)
        self.vel_field = vel_field

    @property
    def scheme_key(self):
        return f"div({self._named(self.vel_field, 'velocity')})"


class PressureGradient:
    """Lazy reference to the CELL-CENTRED gradient an implicit solve stored on ``p``.

    ``p_field.grad[lev]`` is a list of per-box JAX arrays.
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
