# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np

from blockamr.schemes.div_schemes import Upwind


class Div:
    """Divergence operator: div(U * phi).

    vel_func(x, y, z, t) -> (u, v, w) evaluated at face centers.
    Stencil computation uses JAX.
    """

    def __init__(self, vel_func, field, coeff=1.0, scheme=None):
        self.vel_func = vel_func
        self.field = field
        self.coeff = coeff
        self.scheme = scheme or Upwind()
        self._name = "Div"

    def __rmul__(self, scalar):
        return Div(self.vel_func, self.field, coeff=self.coeff * scalar, scheme=self.scheme)

    def _face_value(self, phi, ng, dims, dim, n, offset, vel_face):
        """Compute face values using the scheme stencil.

        offset: 0 for right face (i+1/2), -1 for left face (i-1/2)
        """
        nx, ny, nz = dims

        # Cells immediately left and right of the face
        slc_l = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
        slc_r = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
        slc_l[dim] = slice(ng + offset, ng + offset + n)
        slc_r[dim] = slice(ng + offset + 1, ng + offset + 1 + n)

        phi_l = phi[tuple(slc_l)]
        phi_r = phi[tuple(slc_r)]

        if self.scheme.stencil_width == 1:
            return self.scheme.face_value(phi_l, phi_r, vel_face)

        # Wide stencil: far-left and far-right cells
        slc_fl = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
        slc_fr = [slice(ng, ng + nx), slice(ng, ng + ny), slice(ng, ng + nz)]
        slc_fl[dim] = slice(ng + offset - 1, ng + offset - 1 + n)
        slc_fr[dim] = slice(ng + offset + 2, ng + offset + 2 + n)

        phi_fl = phi[tuple(slc_fl)]
        phi_fr = phi[tuple(slc_fr)]

        return self.scheme.face_value(phi_fl, phi_l, phi_r, phi_fr, vel_face)

    def compute(self, patch, t):
        """Compute divergence on a single patch.

        Returns a JAX array on the valid region.
        """
        ng = patch.ngrow
        dx = patch.geom.cell_size()
        lo = patch.box.small_end()
        prob_lo = patch.geom.prob_lo()

        # Extract phi from grown array (includes ghost cells) as JAX array
        phi = jnp.asarray(patch.grown_arr[:, :, :, 0])

        # Valid region dimensions
        nx, ny, nz = patch.valid_arr.shape[:3]
        dims = (nx, ny, nz)

        # Build cell-center coordinates for the valid region
        xcc = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ycc = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zcc = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])

        result = jnp.zeros((nx, ny, nz))

        for dim, d in enumerate(dx):
            n = dims[dim]

            # Right face coordinates (i+1/2)
            face_coords = list(jnp.meshgrid(xcc, ycc, zcc, indexing="ij"))
            face_coords[dim] = face_coords[dim] + 0.5 * d

            u, v, w = self.vel_func(face_coords[0], face_coords[1], face_coords[2], t)
            vel_right = [u, v, w][dim]

            phi_face_right = self._face_value(phi, ng, dims, dim, n, 0, vel_right)

            # Left face coordinates (i-1/2)
            face_coords_l = list(jnp.meshgrid(xcc, ycc, zcc, indexing="ij"))
            face_coords_l[dim] = face_coords_l[dim] - 0.5 * d

            u_l, v_l, w_l = self.vel_func(
                face_coords_l[0], face_coords_l[1], face_coords_l[2], t
            )
            vel_left = [u_l, v_l, w_l][dim]

            phi_face_left = self._face_value(phi, ng, dims, dim, n, -1, vel_left)

            # Flux divergence: (F_right - F_left) / dx
            result = result + (vel_right * phi_face_right - vel_left * phi_face_left) / d

        return result
