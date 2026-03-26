# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Chorin pressure projection for incompressible flow.

Uses batch array access (arrays()/grown_arrays()) to avoid nested MFIters.
"""

import numpy as np
import jax.numpy as jnp
import neon.blockamr as blockamr


def cell_to_face(vel_mfs, face_mfs):
    """Interpolate cell-centred velocity to face centres (linear average).

    For face d: f[i+1/2] = 0.5 * (c[i] + c[i+1]).
    """
    for d in range(3):
        _interp_to_face(vel_mfs[d], face_mfs[d], d)


def _interp_to_face(cell_mf, face_mf, d):
    """Average cell_mf to face_mf along direction d.

    Uses grown_arrays()/arrays() to avoid nested MFIters.
    For direction d with N cells and ng ghosts:
      grown array has N+2*ng entries.
      Face array has N+1 entries along d, N along other axes.
      f[j] = 0.5*(c[j] + c[j+1]) for j = 0..N, using the grown array
      directly so ghost cells provide the boundary stencil.
    """
    ng = cell_mf.n_grow()
    cell_arrs = cell_mf.grown_arrays()
    face_arrs = face_mf.arrays()

    results = []
    for bi in range(len(face_arrs)):
        c = cell_arrs[bi][:, :, :, 0]
        nf = [int(face_arrs[bi].shape[ax]) for ax in range(3)]

        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        for ax in range(3):
            if ax == d:
                # nf[d] = N+1 face values; need cells from ng-1 to ng+N-1 and ng to ng+N
                sl_lo[ax] = slice(ng - 1, ng - 1 + nf[ax])
                sl_hi[ax] = slice(ng, ng + nf[ax])
            else:
                # nf[ax] = N cell values; take interior only
                sl_lo[ax] = slice(ng, ng + nf[ax])
                sl_hi[ax] = slice(ng, ng + nf[ax])

        results.append(0.5 * (c[tuple(sl_lo)] + c[tuple(sl_hi)]))

    face_mf.copy_arrays(results)


def divergence_arrays(face_mfs, geom):
    """Compute cell-centred divergence from face velocities.

    Returns list of JAX arrays, one per box, shape (nx, ny, nz).
    """
    dx = geom.cell_size()
    face_arrs = [fm.arrays() for fm in face_mfs]
    n_boxes = len(face_arrs[0])

    results = []
    for bi in range(n_boxes):
        div_val = None
        for d in range(3):
            farr = face_arrs[d][bi][:, :, :, 0]
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[d] = slice(0, -1)
            sl_hi[d] = slice(1, None)
            diff = (farr[tuple(sl_hi)] - farr[tuple(sl_lo)]) / dx[d]
            div_val = diff if div_val is None else div_val + diff
        results.append(div_val)
    return results


def face_grad_to_cell_arrays(gx_mf, gy_mf, gz_mf):
    """Interpolate face-centred gradient to cell centres.

    Returns list of [gx, gy, gz] per box, each a JAX array.
    """
    grad_arrs = [gm.arrays() for gm in [gx_mf, gy_mf, gz_mf]]
    n_boxes = len(grad_arrs[0])
    results = []
    for bi in range(n_boxes):
        grads = []
        for d in range(3):
            farr = grad_arrs[d][bi][:, :, :, 0]
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[d] = slice(0, -1)
            sl_hi[d] = slice(1, None)
            grads.append(0.5 * (farr[tuple(sl_lo)] + farr[tuple(sl_hi)]))
        results.append(grads)
    return results


class Projector:
    """Cell-centred approximate projection.

    Solves:  del^2(p) = (1/dt) * div(u*)
    Corrects: u = u* - dt * grad(p)
    """

    def __init__(self, mesh, geom, dt):
        self.mesh = mesh
        self.geom = geom
        self.dt = dt

        ba = mesh.box_array(0)
        dm = mesh.dm(0)
        is_per = geom.is_periodic()

        lo_bc = [blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
                 for d in range(3)]

        self._lp = blockamr.MLPoisson(geom, ba, dm)
        self._lp.set_domain_bc(lo_bc, lo_bc[:])
        self._lp.set_level_bc(0, None)

        self._mlmg = blockamr.MLMG(self._lp)
        self._mlmg.set_verbose(0)
        self._mlmg.set_max_iter(200)
        self._mlmg.set_bottom_verbose(0)

        self._phi = blockamr.MultiFab(ba, dm, 1, 1)
        self._rhs = blockamr.MultiFab(ba, dm, 1, 0)

        dom = geom.domain()
        self._grad = _make_face_mfs(dom, dm, 0)
        self._face_vel = _make_face_mfs(dom, dm, 0)

    def project(self, vel_mfs):
        """Project velocity [u, v, w] MultiFabs to divergence-free. In-place.

        Uses a MAC-style approach:
        1. Interpolate cell velocity to faces
        2. Compute div(u_face)
        3. Solve del^2(p) = div(u_face) / dt
        4. Correct face velocities: u_face -= dt * grad(p)
        5. Average corrected face velocities back to cell centres
        """
        dt = self.dt

        # 1. Interpolate cell velocity to faces
        cell_to_face(vel_mfs, self._face_vel)

        # 2. RHS = (1/dt) * div(u_face)
        div_arrs = divergence_arrays(self._face_vel, self.geom)
        self._rhs.copy_arrays([d / dt for d in div_arrs])

        # 3. Zero initial guess (valid region only — arrays() returns grown shape)
        ng = self._phi.n_grow()
        self._phi.copy_arrays([
            jnp.zeros(tuple(int(a.shape[ax]) - 2 * ng for ax in range(3)))
            for a in self._phi.arrays()
        ])

        # 4. Solve
        self._mlmg.solve(self._phi, self._rhs, 1e-10, 1e-12)

        # 5. Get face gradient and correct face velocities
        self._mlmg.get_grad_solution(*self._grad)
        for d in range(3):
            face_arrs = self._face_vel[d].arrays()
            grad_arrs = self._grad[d].arrays()
            corrected = []
            for bi in range(len(face_arrs)):
                corrected.append(
                    face_arrs[bi][:, :, :, 0] - dt * grad_arrs[bi][:, :, :, 0]
                )
            self._face_vel[d].copy_arrays(corrected)

        # 6. Average corrected face velocities back to cell centres
        #    u_cell[i] = 0.5 * (u_face[i] + u_face[i+1])
        for d in range(3):
            face_arrs = self._face_vel[d].arrays()
            corrected_cells = []
            for bi in range(len(face_arrs)):
                f = face_arrs[bi][:, :, :, 0]
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[d] = slice(0, -1)
                sl_hi[d] = slice(1, None)
                corrected_cells.append(
                    0.5 * (f[tuple(sl_lo)] + f[tuple(sl_hi)])
                )
            vel_mfs[d].copy_arrays(corrected_cells)


def _make_face_mfs(dom, dm, ngrow):
    """Create 3 face-centred MultiFabs."""
    mfs = []
    for d in range(3):
        fb = blockamr.Box(dom.small_end(), dom.big_end())
        fb.surrounding_nodes(d)
        fba = blockamr.BoxArray(fb)
        fba.max_size(32)
        mfs.append(blockamr.MultiFab(fba, dm, 1, ngrow))
    return mfs


# ---------------------------------------------------------------------------
# Nodal projection — JAX stencils + AMReX MLNodeLaplacian solver
# ---------------------------------------------------------------------------

import jax


@jax.jit
def nodal_divergence(u, v, w, dx, dy, dz):
    """Cell-centred velocity -> nodal divergence (interior nodes only).

    u, v, w: JAX arrays of shape (nx, ny, nz) — cell-centred velocity components.
    Returns: shape (nx+1, ny+1, nz+1) — nodal divergence, zero on boundary nodes.

    The stencil at each interior node (i,j,k) averages the 4 surrounding
    face-normal differences, matching MLNodeLaplacian's discretisation.
    """
    # du/dx at cell interfaces, then average to nodes over transverse directions
    dudx = (u[1:, :, :] - u[:-1, :, :]) / dx          # (nx-1, ny, nz)
    dudx_nd = 0.25 * (dudx[:, 1:, 1:] + dudx[:, :-1, 1:]
                     + dudx[:, 1:, :-1] + dudx[:, :-1, :-1])  # (nx-1, ny-1, nz-1)

    dvdy = (v[:, 1:, :] - v[:, :-1, :]) / dy
    dvdy_nd = 0.25 * (dvdy[1:, :, 1:] + dvdy[:-1, :, 1:]
                     + dvdy[1:, :, :-1] + dvdy[:-1, :, :-1])

    dwdz = (w[:, :, 1:] - w[:, :, :-1]) / dz
    dwdz_nd = 0.25 * (dwdz[1:, 1:, :] + dwdz[:-1, 1:, :]
                     + dwdz[1:, :-1, :] + dwdz[:-1, :-1, :])

    interior = dudx_nd + dvdy_nd + dwdz_nd  # (nx-1, ny-1, nz-1)
    return jnp.pad(interior, 1)              # (nx+1, ny+1, nz+1), boundary = 0


@jax.jit
def nodal_gradient(phi, dx, dy, dz):
    """Nodal phi -> cell-centred gradient. Adjoint of nodal_divergence.

    phi: JAX array of shape (nx+1, ny+1, nz+1) — nodal potential.
    Returns: (gx, gy, gz) each of shape (nx, ny, nz) — cell-centred gradient.

    At each cell (i,j,k), averages the 4 surrounding nodal differences.
    """
    dphidx = (phi[1:, :, :] - phi[:-1, :, :]) / dx     # (nx, ny+1, nz+1)
    gx = 0.25 * (dphidx[:, 1:, 1:] + dphidx[:, :-1, 1:]
                + dphidx[:, 1:, :-1] + dphidx[:, :-1, :-1])  # (nx, ny, nz)

    dphidy = (phi[:, 1:, :] - phi[:, :-1, :]) / dy
    gy = 0.25 * (dphidy[1:, :, 1:] + dphidy[:-1, :, 1:]
                + dphidy[1:, :, :-1] + dphidy[:-1, :, :-1])

    dphidz = (phi[:, :, 1:] - phi[:, :, :-1]) / dz
    gz = 0.25 * (dphidz[1:, 1:, :] + dphidz[:-1, 1:, :]
                + dphidz[1:, :-1, :] + dphidz[:-1, :-1, :])

    return gx, gy, gz


class NodalProjector:
    """Nodal pressure projection for incompressible flow.

    Uses AMReX's MLNodeLaplacian + MLMG for the full projection:
    - compDivergence: compute RHS = div(u*) using AMReX's exact nodal stencil
    - MLMG.solve: solve div(sigma * grad(phi)) = RHS
    - MLMG.getFluxes: compute cell-centred correction = -sigma * grad(phi)
    - Velocity correction done in JAX: u += flux

    This ensures the divergence/gradient stencils are exactly consistent
    with the Poisson operator, guaranteeing discrete div-free velocity.
    """

    def __init__(self, mesh, geom, dt, rho=1.0):
        self.sigma = dt / rho
        self.geom = geom

        ba = mesh.box_array(0)
        dm = mesh.dm(0)
        is_per = geom.is_periodic()

        # Nodal Poisson operator
        self._lp = blockamr.MLNodeLaplacian(
            geom, ba, dm, blockamr.LPInfo(), self.sigma)
        lo_bc = [blockamr.LinOpBCType.Periodic if is_per[d]
                 else blockamr.LinOpBCType.Neumann for d in range(3)]
        self._lp.set_domain_bc(lo_bc, lo_bc[:])

        self._mlmg = blockamr.MLMG(self._lp)
        self._mlmg.set_verbose(0)
        self._mlmg.set_max_iter(200)
        self._mlmg.set_bottom_verbose(0)

        # Scratch MultiFabs
        dom = geom.domain()
        lo = dom.small_end()
        hi = dom.big_end()
        N = [hi[d] - lo[d] + 1 for d in range(3)]

        # Nodal phi and rhs
        nodal_box = blockamr.Box(lo, [hi[0] + 1, hi[1] + 1, hi[2] + 1])
        nodal_ba = blockamr.BoxArray(nodal_box)
        nodal_ba.max_size(max(N[0] + 1, 32))
        self._phi = blockamr.MultiFab(nodal_ba, dm, 1, 1)
        self._rhs = blockamr.MultiFab(nodal_ba, dm, 1, 0)

        # Cell-centred velocity (ncomp=3, used by compDivergence)
        self._vel3 = blockamr.MultiFab(ba, dm, 3, 1)

        # Cell-centred fluxes (ncomp=3, returned by getFluxes)
        self._fluxes = blockamr.MultiFab(ba, dm, 3, 0)

    def project(self, vel_mfs):
        """Project velocity [u, v, w] MultiFabs to divergence-free. In-place.

        Steps:
        1. Pack u, v, w into ncomp=3 MultiFab
        2. compDivergence → nodal RHS (AMReX stencil)
        3. MLMG.solve → nodal phi
        4. MLMG.getFluxes → cell-centred correction (= -sigma * grad(phi))
        5. vel += flux (JAX)
        """
        ng = vel_mfs[0].n_grow()

        # 1. Pack velocity into ncomp=3 MultiFab including ghost cells.
        #    vel_mfs[d] have ngrow=1 with ghosts already filled by the caller.
        #    We write the full grown region (valid+ghosts) into _vel3.
        u_grown = vel_mfs[0].grown_arrays()[0][:, :, :, 0]
        v_grown = vel_mfs[1].grown_arrays()[0][:, :, :, 0]
        w_grown = vel_mfs[2].grown_arrays()[0][:, :, :, 0]
        n = [int(u_grown.shape[ax]) - 2 * ng for ax in range(3)]
        sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))

        # Stack into (nx_grown, ny_grown, nz_grown, 3) and write full FAB
        grown_3 = np.asfortranarray(np.array(
            jnp.stack([u_grown, v_grown, w_grown], axis=-1)))
        for mfi in blockamr.MFIterator(self._vel3):
            self._vel3.copy_grown_from(mfi, grown_3)

        # 2. Compute RHS = div(vel) using AMReX's exact nodal stencil (C++)
        self._lp.comp_divergence(self._rhs, self._vel3)

        # 3. Zero initial guess for phi
        ng_phi = self._phi.n_grow()
        phi_valid_shape = tuple(int(self._phi.arrays()[0].shape[ax]) - 2 * ng_phi
                                for ax in range(3))
        self._phi.copy_arrays([jnp.zeros(phi_valid_shape)])

        # 4. Solve: div(sigma * grad(phi)) = RHS (AMReX MLMG, C++)
        self._mlmg.solve(self._phi, self._rhs, 1e-10, 1e-12)

        # 5. Get cell-centred fluxes = -sigma * grad(phi) (AMReX, C++)
        self._mlmg.get_fluxes(self._fluxes)

        # 6. Correct velocity: vel += flux (JAX on valid region)
        flux = self._fluxes.arrays()[0]  # (nx, ny, nz, 3)
        for d in range(3):
            v_grown = vel_mfs[d].arrays()[0][:, :, :, 0]
            v_valid = v_grown[sl]
            vel_mfs[d].copy_arrays([v_valid + flux[:, :, :, d]])

    def project_cellfield(self, U_field):
        """Project a CellField(ncomp=3) to divergence-free. In-place.

        Accepts a single CellField with 3 velocity components instead of
        a list of 3 separate MultiFabs.
        """
        mf = U_field.mf[0]
        ng = mf.n_grow()

        # 1. Copy velocity (with ghosts) into _vel3
        grown = mf.grown_arrays()[0]  # (nx+2ng, ny+2ng, nz+2ng, 3)
        grown_np = np.asfortranarray(np.array(grown))
        for mfi in blockamr.MFIterator(self._vel3):
            self._vel3.copy_grown_from(mfi, grown_np)

        # 2. compDivergence → nodal RHS
        self._lp.comp_divergence(self._rhs, self._vel3)

        # 3. Zero initial guess
        ng_phi = self._phi.n_grow()
        phi_valid_shape = tuple(int(self._phi.arrays()[0].shape[ax]) - 2 * ng_phi
                                for ax in range(3))
        self._phi.copy_arrays([jnp.zeros(phi_valid_shape)])

        # 4. Solve
        self._mlmg.solve(self._phi, self._rhs, 1e-10, 1e-12)

        # 5. Get fluxes
        self._mlmg.get_fluxes(self._fluxes)

        # 6. Correct: U += flux (all 3 components at once)
        n = [int(grown.shape[ax]) - 2 * ng for ax in range(3)]
        sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))
        U_valid = mf.arrays()[0][sl[0], sl[1], sl[2], :]  # (nx, ny, nz, 3)
        flux = self._fluxes.arrays()[0]  # (nx, ny, nz, 3)
        corrected = jnp.stack(
            [U_valid[:, :, :, c] + flux[:, :, :, c] for c in range(3)], axis=-1)
        mf.copy_arrays([corrected])
