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

    def compute(self, patch, t):
        """Compute gradient on the valid region. Returns (nx, ny, nz, 3)."""
        ng = patch.ngrow
        dx = patch.geom.cell_size()

        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])

        nx, ny, nz = patch.valid_arr.shape[:3]
        dims = [nx, ny, nz]
        result = jnp.zeros((nx, ny, nz, 3))

        for dim in range(3):
            n = dims[dim]
            d = dx[dim]

            slc_r = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
            slc_l = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
            slc_r[dim] = slice(ng + 1, ng + 1 + n)
            slc_l[dim] = slice(ng - 1, ng - 1 + n)

            dphi = self.scheme.face_value(phi[tuple(slc_l)], phi[tuple(slc_r)], d)
            result = result.at[:, :, :, dim].set(dphi)

        return result
