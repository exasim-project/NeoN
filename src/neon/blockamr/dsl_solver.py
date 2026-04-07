# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""DSL-based incompressible Navier-Stokes solver.

Uses the OpenFOAM-style DSL syntax with two-step projection
(MAC + nodal) matching IAMReX/incflo:

    1. interpolate(U, phi)
    2. MAC project phi → div-free face fluxes
    3. ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    4. laplacian(dt, p) == div(U*)
    5. U -= dt * grad(p)
"""

import jax.numpy as jnp

import neon.blockamr as blockamr
from .field import CellField, FaceField
from .bc import VectorBC, fixedValue, noSlip
from .fillpatch import FillPatchWithBC
from .dsl import exp, imp, solve
from .operators.interpolate import interpolate
from .operators.correct import correct


class DSLIncompressibleSolver:
    """Incompressible Navier-Stokes solver using the DSL.

    Works with both single-level Mesh and multi-level AmrMesh.
    Uses two-step projection (MAC + nodal) as in IAMReX/incflo.

    Parameters
    ----------
    mesh : Mesh or AmrMesh
    nu : float
        Kinematic viscosity.
    dt : float
    U_bc : VectorBC, optional
        Boundary conditions for the velocity field.
        Mutually exclusive with *fill_patch*.
    schemes_p : dict, optional
        Solver settings for the pressure Poisson equation.
    fill_patch : object, optional
        Fill-patch strategy for the velocity field (e.g.
        ``FillPatchCellConservative()`` for fully periodic domains).
        Mutually exclusive with *U_bc*.
    """

    def __init__(self, mesh, nu, dt, U_bc=None, schemes_p=None, fill_patch=None,
                 div_scheme=None, cfl=None):
        if U_bc is not None and fill_patch is not None:
            raise ValueError("Specify either U_bc or fill_patch, not both.")
        if U_bc is None and fill_patch is None:
            raise ValueError("One of U_bc or fill_patch must be provided.")

        self.mesh = mesh
        self.nu = nu
        self.dt = dt
        self._t = 0.0
        self._cfl = cfl
        self._dx = mesh.geom(0).cell_size()
        self._div_scheme = div_scheme

        # Derive ngrow from the widest stencil across all operators
        # (div scheme + laplacian scheme). Not hardcoded.
        from .schemes.laplacian_schemes import CentralDiffLaplacian
        from .schemes.div_schemes import Upwind
        div_sw = getattr(div_scheme, 'stencil_width', Upwind().stencil_width)
        lap_sw = CentralDiffLaplacian().stencil_width
        ngrow = max(div_sw, lap_sw)

        fp = fill_patch if fill_patch is not None else FillPatchWithBC(U_bc)
        self.U = CellField(
            mesh, ncomp=3, ngrow=ngrow, name="U",
            fill_patch=fp,
        )
        self.p = CellField(mesh, ncomp=1, ngrow=0, name="p")
        self.phi = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")

        self._nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)
        self._schemes_p = schemes_p or {
            "rtol": 1e-10, "atol": 1e-12, "max_iter": 200, "verbose": 0,
        }
        self._mac_solver = None

    @property
    def time(self):
        return self._t

    def step(self):
        """Advance one time step using the DSL.

        Two-step projection matching IAMReX/incflo:

        1. Fill BCs on U
        2. Interpolate U → phi (not div-free)
        3. MAC projection: make phi div-free (MLABecLaplacian + face-centred getFluxes)
        4. Momentum predictor with div-free phi
        5. Fill BCs on U*
        6. Nodal pressure solve: laplacian(dt, p) = div(U*)
        7. Correct U: U^{n+1} = U* - dt * grad(p)
        """
        dt = self.dt
        U = self.U
        p = self.p
        phi = self.phi
        t = self._t
        mesh = self.mesh
        n_levels = mesh.n_levels()

        # 1. Fill BCs on U
        for lev in range(n_levels):
            U.fill_patch(lev, t)

        # 2. Interpolate U to face fluxes (not div-free)
        interpolate(U, phi)

        # 3. MAC projection: make phi divergence-free
        self._mac_project(phi)

        # 4. Momentum predictor with div-free phi
        solve(
            exp.ddt(U) + exp.div(phi, U, scheme=self._div_scheme)
            - exp.laplacian(self._nu_func, U),
            t, dt,
        )

        # 5. Fill BCs on U*
        for lev in range(n_levels):
            U.fill_patch(lev, t)

        # 6. Nodal pressure solve: laplacian(dt, p) = div(U*)
        solve(imp.laplacian(dt, p) == exp.div(U), schemes=self._schemes_p)

        # 7. Correct U: U^{n+1} = U* - dt * grad(p)
        correct(U, -dt * exp.grad(p))

        self._t += dt

        # Adaptive time stepping
        if self._cfl is not None:
            max_vel = self._max_velocity()
            if max_vel > 1e-12:
                finest = mesh.n_levels() - 1
                dx_fine = mesh.geom(finest).cell_size()
                self.dt = self._cfl * min(dx_fine) / max_vel

    # ------------------------------------------------------------------
    # MAC projection
    # ------------------------------------------------------------------

    def _mac_project(self, phi):
        """Project face fluxes to be divergence-free using MLABecLaplacian.

        Solves: div(beta * grad(p_mac)) = div(phi)
        Corrects: phi_f -= beta * grad_f(p_mac)

        Uses MLABecLaplacian with alpha=0, beta=1 (reduces to Laplacian
        for constant density). getFluxes returns face-centred gradients,
        which are the exact adjoint of the face divergence operator.
        """
        mesh = self.mesh
        for lev in range(mesh.n_levels()):
            self._mac_project_level(phi, lev)

    def _mac_project_level(self, phi, lev):
        """MAC projection for one level."""
        mesh = self.mesh
        geom = mesh.geom(lev)
        ba = mesh.box_array(lev)
        dm = mesh.dm(lev)
        dx = geom.cell_size()

        cache = self._ensure_mac_cache(lev)

        # 1. Compute RHS = -div(phi) from face fluxes (JAX)
        # MLABecLaplacian solves (alpha*a - beta*div(b*grad))phi = rhs
        # with alpha=0, beta=1: -div(grad(p)) = rhs
        # We want div(grad(p)) = div(phi), so rhs = -div(phi)
        rhs_arrs = [-arr for arr in self._face_divergence(phi, lev)]
        cache['rhs_mf'].copy_arrays(rhs_arrs)

        # 2. Zero initial guess
        cache['phi_mf'].set_val(0.0)

        # 3. Solve: div(beta * grad(p_mac)) = div(phi)
        cfg = self._schemes_p
        cache['mlmg'].solve(
            cache['phi_mf'], cache['rhs_mf'],
            cfg.get("rtol", 1e-10), cfg.get("atol", 1e-12),
        )

        # 4. Get face-centred fluxes = -beta * grad(p_mac)
        cache['mlmg'].get_fluxes(cache['flux_x'], cache['flux_y'], cache['flux_z'])

        # 5. Correct phi: phi_f += flux_f (flux = -beta*grad, so phi -= beta*grad)
        for d in range(3):
            face_mf = phi[lev][d].mf
            face_ng = face_mf.n_grow()
            face_arrs = face_mf.arrays()
            flux_arrs = cache[f'flux_{"xyz"[d]}'].arrays()

            results = []
            for bi in range(len(face_arrs)):
                f = face_arrs[bi][:, :, :, 0]
                fl = flux_arrs[bi][:, :, :, 0]
                # flux has ngrow=0, face has ngrow=face_ng
                if face_ng > 0:
                    nf = [int(face_arrs[bi].shape[ax]) - 2 * face_ng for ax in range(3)]
                    sl = tuple(slice(face_ng, face_ng + nf[ax]) for ax in range(3))
                    results.append(f[sl] + fl)
                else:
                    results.append(f + fl)

            face_mf.copy_arrays(results)
            face_mf.fill_boundary(geom)

    def _face_divergence(self, phi, lev):
        """Compute cell-centred divergence from face fluxes (JAX).

        Returns list of per-box arrays (nx, ny, nz).
        """
        dx = self.mesh.geom(lev).cell_size()
        face_arrs = [phi[lev][d].mf.arrays() for d in range(3)]
        face_ngs = [phi[lev][d].mf.n_grow() for d in range(3)]
        n_boxes = len(face_arrs[0])

        results = []
        for bi in range(n_boxes):
            div_val = None
            for d in range(3):
                f = face_arrs[d][bi][:, :, :, 0]
                ng = face_ngs[d]
                # Valid cell count: face valid count - 1 in normal dir, same in others
                nf = [int(f.shape[ax]) - 2 * ng for ax in range(3)]
                nc = list(nf)
                nc[d] -= 1  # one fewer cell than faces in normal direction

                sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
                sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
                sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
                sl_lo[d] = slice(ng, ng + nc[d])
                contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
                div_val = contrib if div_val is None else div_val + contrib
            results.append(div_val)
        return results

    def _ensure_mac_cache(self, lev):
        """Build or return cached MAC solver objects for one level."""
        if self._mac_solver is not None and self._mac_solver.get('lev') == lev:
            # Rebind face data but reuse operator/solver
            return self._mac_solver

        mesh = self.mesh
        geom = mesh.geom(lev)
        ba = mesh.box_array(lev)
        dm = mesh.dm(lev)
        is_per = geom.is_periodic()

        # MLABecLaplacian: (alpha*a - beta*div(b*grad)) phi
        # For MAC: alpha=0, beta=1, b=1 → -div(grad(phi)) = RHS
        lp = blockamr.MLABecLaplacian(geom, ba, dm, blockamr.LPInfo())
        lo_bc = [blockamr.LinOpBCType.Periodic if is_per[d]
                 else blockamr.LinOpBCType.Neumann for d in range(3)]
        lp.set_domain_bc(lo_bc, lo_bc[:])
        lp.set_level_bc(0, None)
        lp.set_scalars(0.0, 1.0)  # alpha=0, beta=1

        # b-coefficients = 1 on all faces
        b_mfs = []
        for d in range(3):
            ba_copy = blockamr.BoxArray(ba)
            ba_copy.surrounding_nodes(d)
            b_mf = blockamr.MultiFab(ba_copy, dm, 1, 0)
            b_mf.set_val(1.0)
            ba_copy.enclosed_cells(d)
            b_mfs.append(b_mf)
        lp.set_b_coeffs(0, b_mfs[0], b_mfs[1], b_mfs[2])

        mlmg = blockamr.MLMG(lp)
        mlmg.set_verbose(0)
        mlmg.set_max_iter(200)
        mlmg.set_bottom_verbose(0)

        # Scratch MultiFabs
        phi_mf = blockamr.MultiFab(ba, dm, 1, 1)
        rhs_mf = blockamr.MultiFab(ba, dm, 1, 0)

        # Face-centred flux MultiFabs for getFluxes output
        flux_mfs = {}
        for d, name in enumerate("xyz"):
            ba_face = blockamr.BoxArray(ba)
            ba_face.surrounding_nodes(d)
            flux_mf = blockamr.MultiFab(ba_face, dm, 1, 0)
            ba_face.enclosed_cells(d)
            flux_mfs[f'flux_{name}'] = flux_mf

        self._mac_solver = {
            'lp': lp,
            'mlmg': mlmg,
            'phi_mf': phi_mf,
            'rhs_mf': rhs_mf,
            'lev': lev,
            'b_mfs': b_mfs,
            **flux_mfs,
        }
        return self._mac_solver

    # ------------------------------------------------------------------
    # Regrid / plotfile / utilities
    # ------------------------------------------------------------------

    def regrid(self, tag):
        """Regrid the AMR mesh. No-op for single-level meshes."""
        from .mesh import AmrMesh
        if isinstance(self.mesh, AmrMesh):
            # Fill ghost cells so tagging stencils have valid data
            for lev in range(self.mesh.n_levels()):
                self.U.fill_patch(lev, self._t)
            self.mesh.regrid(self._t, tag=tag)
            # Invalidate solver caches — grids changed
            if hasattr(self.p, '_imp_solver'):
                del self.p._imp_solver
            self._mac_solver = None
            from .dsl.solve import BF
            mesh = self.mesh
            total_cells = 0
            print(f"  Regrid: {mesh.n_levels()} levels")
            for lev in range(mesh.n_levels()):
                mf = self.U.mf[lev]
                if mf is None:
                    continue
                ng = mf.n_grow()
                lev_cells = sum(
                    (m[1]-2*ng)*(m[2]-2*ng)*(m[3]-2*ng)
                    for m in mf.fab_metadata())
                total_cells += lev_cells
                nboxes = len(mf.fab_metadata())
                layout = blockamr.build_tile_layout(mf, BF)
                buf_size = mf.contiguous_array().size
                padded_cap = self.U._padded_cap[lev]
                print(f"    lev {lev}: {lev_cells:,} cells, {nboxes} boxes, "
                      f"tiles={layout.n_tiles} (padded={layout.n_tiles_padded}), bf={BF}\n"
                      f"           buf={buf_size:,} (cap={padded_cap:,})")
            print(f"    total: {total_cells:,} cells")

    def write_plotfile(self, name, fields=None):
        """Write a plotfile. Works for both single-level and AMR."""
        import os, shutil
        if os.path.exists(name):
            shutil.rmtree(name)

        if fields is None:
            fields = [self.U]

        mesh = self.mesh
        n_levels = mesh.n_levels()

        varnames = []
        for f in fields:
            if f.ncomp == 1:
                varnames.append(f.name)
            else:
                _suffixes = ["_x", "_y", "_z"]
                varnames.extend([f"{f.name}{_suffixes[c]}" for c in range(f.ncomp)])

        if len(fields) == 1:
            mfs = [fields[0].mf[lev] for lev in range(n_levels)]
        else:
            total_ncomp = sum(f.ncomp for f in fields)
            mfs = []
            for lev in range(n_levels):
                combined = blockamr.MultiFab(
                    mesh.box_array(lev), mesh.dm(lev), total_ncomp, 0)
                n_boxes = len(fields[0].mf[lev].arrays())
                results = []
                for bi in range(n_boxes):
                    parts = []
                    for f in fields:
                        arr = f.mf[lev].arrays()[bi]
                        ng = f.mf[lev].n_grow()
                        n = [int(arr.shape[ax]) - 2 * ng for ax in range(3)]
                        sl = tuple(slice(ng, ng + n[ax]) for ax in range(3))
                        parts.append(arr[sl[0], sl[1], sl[2], :])
                    results.append(jnp.concatenate(parts, axis=-1))
                combined.copy_arrays(results)
                mfs.append(combined)

        if n_levels == 1:
            blockamr.write_single_level_plotfile(
                name, mfs[0], varnames, mesh.geom(0), self._t, 0,
            )
        else:
            blockamr.write_multilevel_plotfile(
                name, n_levels, mfs, varnames,
                [mesh.geom(lev) for lev in range(n_levels)],
                self._t, [0] * n_levels,
                [mesh.ref_ratio(lev) for lev in range(n_levels - 1)],
            )

    def _max_velocity(self):
        """Return max |U| across all levels (conservative — includes ghost cells)."""
        max_sq = jnp.float32(0.0)
        for lev in range(self.mesh.n_levels()):
            mf = self.U.mf[lev]
            if mf is None:
                continue
            padded = self.U._padded_cap[lev] if hasattr(self.U, '_padded_cap') else 0
            flat = mf.contiguous_array(padded)
            meta = mf.fab_metadata()
            _, Nx, Ny, Nz, nc = meta[0]
            M = Nx * Ny * Nz
            n_boxes = len(meta)
            if all(m[1] * m[2] * m[3] == M for m in meta):
                all_data = flat[: n_boxes * nc * M].reshape(n_boxes, nc, M)
                mag_sq = jnp.sum(all_data**2, axis=1)
                max_sq = jnp.maximum(max_sq, jnp.max(mag_sq))
            else:
                for offset, bNx, bNy, bNz, bnc in meta:
                    bM = bNx * bNy * bNz
                    box_data = flat[offset : offset + bM * bnc].reshape(bnc, bM)
                    mag_sq = jnp.sum(box_data**2, axis=0)
                    max_sq = jnp.maximum(max_sq, jnp.max(mag_sq))
        return float(jnp.sqrt(max_sq))
