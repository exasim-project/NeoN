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

    def compute(self, patch, t):
        """Compute div(gamma * grad(phi)) on the valid region."""
        ng = patch.ngrow
        dx = patch.geom.cell_size()
        lo = patch.box.small_end()
        prob_lo = patch.geom.prob_lo()

        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])

        nx, ny, nz = patch.valid_arr.shape[:3]

        # Cell-center coordinates for the grown (ghost-inclusive) region
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

        # Evaluate gamma on the grown region
        X, Y, Z = jnp.meshgrid(cc[0], cc[1], cc[2], indexing="ij")
        gamma = self.gamma_func(X, Y, Z, t)

        result = jnp.zeros((nx, ny, nz))

        for dim in range(3):
            d = dx[dim]
            n = dims[dim]

            # Slices for center, right, and left cells in the grown array
            slc_c = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
            slc_r = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
            slc_l = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
            slc_r[dim] = slice(ng + 1, ng + 1 + n)
            slc_l[dim] = slice(ng - 1, ng - 1 + n)

            phi_c = phi[tuple(slc_c)]
            phi_r = phi[tuple(slc_r)]
            phi_l = phi[tuple(slc_l)]

            gamma_c = gamma[tuple(slc_c)]
            gamma_r = gamma[tuple(slc_r)]
            gamma_l = gamma[tuple(slc_l)]

            # Face-averaged gamma via scheme stencil
            gamma_right = self.scheme.face_value(gamma_c, gamma_r)
            gamma_left = self.scheme.face_value(gamma_l, gamma_c)

            # Central difference: (gamma_R*(phi_R - phi_C) - gamma_L*(phi_C - phi_L)) / dx^2
            result = result + (
                gamma_right * (phi_r - phi_c) - gamma_left * (phi_c - phi_l)
            ) / (d * d)

        return result
