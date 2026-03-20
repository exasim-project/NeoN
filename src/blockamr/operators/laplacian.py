# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian


class Laplacian:
    """Laplacian operator: div(gamma * grad(phi)).

    gamma_func(x, y, z, t) -> scalar_array evaluated at cell centers.
    """

    def __init__(self, gamma_func, field, coeff=1.0, scheme=None):
        self.gamma_func = gamma_func
        self.field = field
        self.coeff = coeff
        self.scheme = scheme or CentralDiffLaplacian()
        self._name = "Laplacian"

    def __rmul__(self, scalar):
        return Laplacian(
            self.gamma_func, self.field, coeff=self.coeff * scalar, scheme=self.scheme
        )

    def build_kernel(self, mfi, t):
        """Return a scheme functor bound to this mfi's gamma."""
        ng = self.field.mf.n_grow()
        dx = self.field.geom.cell_size()
        lo = mfi.valid_box().small_end()
        prob_lo = self.field.geom.prob_lo()
        bx = mfi.valid_box()
        lo_v = bx.small_end()
        hi_v = bx.big_end()
        nx = hi_v[0] - lo_v[0] + 1
        ny = hi_v[1] - lo_v[1] + 1
        nz = hi_v[2] - lo_v[2] + 1

        dims = [nx, ny, nz]
        cc = []
        for dim in range(3):
            n = dims[dim]
            cc.append(
                jnp.array(
                    [
                        prob_lo[dim] + (lo[dim] - ng + i + 0.5) * dx[dim]
                        for i in range(n + 2 * ng)
                    ]
                )
            )

        X, Y, Z = jnp.meshgrid(cc[0], cc[1], cc[2], indexing="ij")
        gamma = self.gamma_func(X, Y, Z, t)
        dh = jnp.array(dx)
        return self.scheme.build_kernel(gamma, dh, coeff=self.coeff)
