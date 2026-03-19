# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import jax.numpy as jnp


class SourceKernel(NamedTuple):
    S: object  # Array
    ng: int
    coeff: float

    def __call__(self, phi):
        phi_valid = phi[self.ng : -self.ng, self.ng : -self.ng, self.ng : -self.ng] if self.ng > 0 else phi
        return self.coeff * self.S * phi_valid


class Source:
    """Pointwise source term: S(x,y,z,t) * phi.

    coeff_func(x, y, z, t) -> scalar_array evaluated at cell centers.
    """

    def __init__(self, coeff_func, field, coeff=1.0):
        self.coeff_func = coeff_func
        self.field = field
        self.coeff = coeff
        self._name = "Source"

    def __rmul__(self, scalar):
        return Source(self.coeff_func, self.field, coeff=self.coeff * scalar)

    def build_kernel(self, mfi, t):
        """Return a SourceKernel functor for this mfi."""
        ng = self.field.mf.n_grow()
        lo = mfi.valid_box().small_end()
        dx = self.field.geom.cell_size()
        prob_lo = self.field.geom.prob_lo()
        valid_arr = self.field.mf.array(mfi)
        nx, ny, nz = valid_arr.shape[:3]

        xcc = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ycc = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zcc = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])

        X, Y, Z = jnp.meshgrid(xcc, ycc, zcc, indexing="ij")
        S = self.coeff_func(X, Y, Z, t)
        return SourceKernel(S=S, ng=ng, coeff=self.coeff)
