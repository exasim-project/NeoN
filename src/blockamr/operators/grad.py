# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from ..schemes.grad_schemes import CentralDiffGrad
from ..dsl.eqterm import EqTerm


class Grad(EqTerm):
    """Gradient operator: (dphi/dx, dphi/dy, dphi/dz).

    Returns shape (nx, ny, nz, 3) — a vector field.
    """

    kind = "spatial"
    _scheme_operator = "grad"
    scheme_key = "grad"

    def __init__(self, field, coeff=1.0, scheme=None):
        super().__init__(field, coeff=coeff, scheme=scheme or CentralDiffGrad())
        self._scheme_explicit = scheme is not None

    def build_kernel(self, mfi, t, lev=0):
        """Return a scheme functor bound to this field's dh."""
        dh = jnp.array(self.field.mesh.geom(lev).cell_size())
        ngrow = self.field.mf[lev].n_grow()
        return self.scheme.build_kernel(dh, coeff=self.coeff, ngrow=ngrow)
