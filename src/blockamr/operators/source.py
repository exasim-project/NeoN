# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp


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

    def compute(self, patch, t):
        """Compute coeff_func * phi on the valid region."""
        lo = patch.box.small_end()
        dx = patch.geom.cell_size()
        prob_lo = patch.geom.prob_lo()

        nx, ny, nz = patch.valid_arr.shape[:3]

        xcc = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ycc = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zcc = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])

        X, Y, Z = jnp.meshgrid(xcc, ycc, zcc, indexing="ij")
        S = self.coeff_func(X, Y, Z, t)

        phi = jnp.asarray(patch.valid_arr[:, :, :, 0])
        return S * phi
