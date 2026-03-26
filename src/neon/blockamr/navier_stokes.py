# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Incompressible Navier-Stokes solver using Chorin projection.

Single-level, constant density, explicit advection + explicit diffusion,
forward Euler time stepping. Uses JAX for on-device computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import neon.blockamr as blockamr
from .bc import BoundaryCondition, fill_ghost_cells
from .projection import (
    NodalProjector, Projector, cell_to_face, divergence_arrays,
    nodal_divergence, _make_face_mfs,
)


@jax.jit(static_argnums=(4, 5, 6, 7))
def _advection_diffusion(phi, face_x, face_y, face_z,
                         nx, ny, nz, ng, dx, nu):
    """Compute -div(u_face * phi) + nu * laplacian(phi).

    phi: 3D JAX array with ghosts (nx+2ng, ny+2ng, nz+2ng)
    face_x/y/z: face-centred arrays (valid region only)
    Returns: (nx, ny, nz) array.
    """
    dims = jnp.array([nx, ny, nz])
    faces = [face_x, face_y, face_z]
    conv = jnp.zeros((nx, ny, nz))
    diff = jnp.zeros((nx, ny, nz))

    for ax in range(3):
        n = dims[ax]
        f = faces[ax]

        # Left and right faces of each cell
        sl_fl = [slice(None)] * 3
        sl_fr = [slice(None)] * 3
        sl_fl[ax] = slice(0, -1)
        sl_fr[ax] = slice(1, None)
        fl = f[tuple(sl_fl)]
        fr = f[tuple(sl_fr)]

        # Cell values: centre, left neighbour, right neighbour
        sl_c = [slice(ng, ng + [nx, ny, nz][a]) for a in range(3)]
        sl_l = list(sl_c)
        sl_r = list(sl_c)
        sl_l[ax] = slice(ng - 1, ng + [nx, ny, nz][ax] - 1)
        sl_r[ax] = slice(ng + 1, ng + [nx, ny, nz][ax] + 1)

        phi_c = phi[tuple(sl_c)]
        phi_l = phi[tuple(sl_l)]
        phi_r = phi[tuple(sl_r)]

        # Upwind convective flux
        F_l = fl * jnp.where(fl >= 0, phi_l, phi_c)
        F_r = fr * jnp.where(fr >= 0, phi_c, phi_r)
        conv = conv + (F_r - F_l) / dx[ax]

        # Central difference diffusion
        diff = diff + (phi_r - 2.0 * phi_c + phi_l) / dx[ax] ** 2

    return -conv + nu * diff


class IncompressibleSolver:
    """Single-level incompressible Navier-Stokes solver.

    Parameters
    ----------
    mesh : Mesh
    geom : Geometry
    nu : float
        Kinematic viscosity.
    u_bc, v_bc, w_bc : BoundaryCondition
    dt : float
    """

    def __init__(self, mesh, geom, nu, u_bc, v_bc, w_bc, dt):
        self.mesh = mesh
        self.geom = geom
        self.nu = nu
        self.dt = dt
        self._bcs = [u_bc, v_bc, w_bc]

        ba = mesh.box_array(0)
        dm = mesh.dm(0)

        self.vel = [blockamr.MultiFab(ba, dm, 1, 1) for _ in range(3)]
        for mf in self.vel:
            _zero_mf(mf)

        dom = geom.domain()
        self._face_vel = _make_face_mfs(dom, dm, 0)
        self._proj = NodalProjector(mesh, geom, dt)

    def step(self):
        """Advance one time step."""
        dt = self.dt
        nu = self.nu
        dx = jnp.array(self.geom.cell_size())

        # 1. Fill ghost cells
        self._fill_bcs()

        # 2. Interpolate to faces
        cell_to_face(self.vel, self._face_vel)

        # 3-5. Compute RHS and update each velocity component
        # Pre-read face arrays (avoids nested MFIters)
        face_arrs = [fm.arrays() for fm in self._face_vel]

        for d in range(3):
            vel_mf = self.vel[d]
            grown = vel_mf.grown_arrays()  # list of 4D JAX arrays
            ng = vel_mf.n_grow()

            results = []
            for bi in range(len(grown)):
                phi = grown[bi][:, :, :, 0]
                valid_shape = [phi.shape[ax] - 2 * ng for ax in range(3)]
                nx, ny, nz = valid_shape

                rhs = _advection_diffusion(
                    phi,
                    face_arrs[0][bi][:, :, :, 0],
                    face_arrs[1][bi][:, :, :, 0],
                    face_arrs[2][bi][:, :, :, 0],
                    nx, ny, nz, ng, dx, nu,
                )

                # Forward Euler
                sl_valid = tuple(slice(ng, ng + n) for n in valid_shape)
                phi_new = phi[sl_valid] + dt * rhs
                results.append(phi_new)

            vel_mf.copy_arrays(results)

        # 6. Fill BCs on u*
        self._fill_bcs()

        # 7. Project
        self._proj.project(self.vel)

        # 8. Fill BCs on corrected velocity
        self._fill_bcs()

    def _fill_bcs(self):
        """Fill ghost cells for all velocity components."""
        for d in range(3):
            self.vel[d].fill_boundary(self.geom)
            fill_ghost_cells(self.vel[d], self.geom, self._bcs[d])

    def max_velocity(self):
        """Return max |u| across all components."""
        max_val = 0.0
        for mf in self.vel:
            for arr in mf.arrays():
                max_val = max(max_val, float(jnp.max(jnp.abs(arr))))
        return max_val

    def divergence_error(self):
        """Return max |div(u)|."""
        self._fill_bcs()
        cell_to_face(self.vel, self._face_vel)
        div_arrs = divergence_arrays(self._face_vel, self.geom)
        return max(float(jnp.max(jnp.abs(d))) for d in div_arrs)


def _zero_mf(mf):
    """Set all values in a MultiFab to zero."""
    zeros = [jnp.zeros_like(a[:, :, :, 0]) for a in mf.arrays()]
    mf.copy_arrays(zeros)
