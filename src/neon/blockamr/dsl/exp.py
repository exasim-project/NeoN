# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from ..operators.ddt import Ddt
from ..operators.div import Div
from ..operators.grad import Grad
from ..operators.laplacian import Laplacian
from ..operators.source import Source


def ddt(field):
    return Ddt(field)


def div(face_fluxes, field, scheme=None):
    return Div(face_fluxes, field, scheme=scheme)


def grad(field, scheme=None):
    return Grad(field, scheme=scheme)


def laplacian(gamma_func, field, scheme=None):
    return Laplacian(gamma_func, field, scheme=scheme)


def source(coeff_func, field):
    return Source(coeff_func, field)
