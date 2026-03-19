# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from blockamr.schemes.grad_schemes import CentralDiffGrad


class Grad:
    """Gradient operator: (dphi/dx, dphi/dy, dphi/dz).

    Returns shape (nx, ny, nz, 3) — a vector field.
    """

    def __init__(self, field, coeff=1.0, scheme=None):
        self.field = field
        self.coeff = coeff
        self.scheme = scheme or CentralDiffGrad()
        self._name = "Grad"

    def __rmul__(self, scalar):
        return Grad(self.field, coeff=self.coeff * scalar, scheme=self.scheme)

    def build_kernel(self, mfi, t):
        """Return a scheme functor bound to this field's dh."""
        dh = jnp.array(self.field.geom.cell_size())
        return self.scheme.build_kernel(dh, coeff=self.coeff)
