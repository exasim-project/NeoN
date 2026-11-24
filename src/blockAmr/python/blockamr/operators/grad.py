# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from ..schemes.grad_schemes import CentralDiffGrad
from ..dsl.eqterm import EqTerm


class Grad(EqTerm):
    """Gradient ``(dphi/dx, dphi/dy, dphi/dz)``, shape (nx, ny, nz, 3)."""

    kind = "spatial"
    _scheme_operator = "grad"
    scheme_key = "grad"
    out_ncomp = 3

    def __init__(self, field, coeff=1.0, scheme=None):
        super().__init__(field, coeff=coeff, scheme=scheme or CentralDiffGrad())
        self._scheme_explicit = scheme is not None
