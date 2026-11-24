# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Explicit DSL operators (cf. OpenFOAM fvc::)."""

from ..operators.ddt import Ddt
from ..operators.div import Div
from ..operators.grad import Grad
from ..operators.laplacian import Laplacian
from ..operators.source import Source


def ddt(field):
    return Ddt(field)


def div(face_fluxes, field):
    """Divergence: ``exp.div(phi, U)`` advective flux.

    The scheme is resolved by name from the equation's ``schemes`` at solve time;
    construct ``Div`` directly to pin a scheme object instead.
    """
    return Div(face_fluxes, field)


def grad(field):
    return Grad(field)


def laplacian(gamma_func, field):
    return Laplacian(gamma_func, field)


def source(coeff_func, field):
    return Source(coeff_func, field)

