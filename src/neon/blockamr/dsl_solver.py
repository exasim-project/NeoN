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
from .fillpatch import FillPatchWithBC
from .dsl import exp, imp
from .dsl.equation import Equation
from .ibm import IBM
from .operators.interpolate import interpolate
from .operators.correct import correct
from .operators.mac_project import mac_project
from .schemes.registry import lookup_scheme
from .schemes.laplacian_schemes import CentralDiffLaplacian
from .schemes.div_schemes import Upwind


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
    schemes : dict, optional
        fvSchemes: discretisation scheme names, bound to UEqn/pEqn at
        construction (e.g. ``{"div(phi,U)": "vanLeer"}``).
    sol_U : dict, optional
        fvSolution.solvers['U'] block, passed to ``UEqn.solve(solution=...)``.
    sol_p : dict, optional
        fvSolution.solvers['p'] block (MLMG rtol/atol/maxIter/bottomSolver),
        passed to ``pEqn.solve(solution=...)`` and to the MAC projection.
    U_bc : VectorBC, optional
        Boundary conditions for the velocity field.
        Mutually exclusive with *fill_patch*.
    fill_patch : object, optional
        Fill-patch strategy for the velocity field (e.g.
        ``FillPatchCellConservative()`` for fully periodic domains).
        Mutually exclusive with *U_bc*.
    cfl : float, optional
        Adaptive time-stepping CFL target.
    """

    def __init__(
        self,
        mesh,
        nu,
        dt,
        *,
        schemes=None,
        sol_U=None,
        sol_p=None,
        U_bc=None,
        fill_patch=None,
        cfl=None,
    ):
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
        self.schemes = schemes or {}

        # Derive ngrow from the widest stencil across all operators
        # (div scheme + laplacian scheme). Not hardcoded.
        div_stencil_scheme = lookup_scheme(self.schemes, ["div(phi,U)"], "div", Upwind())
        div_sw = div_stencil_scheme.stencil_width
        lap_sw = CentralDiffLaplacian().stencil_width
        ngrow = max(div_sw, lap_sw)

        fp = fill_patch if fill_patch is not None else FillPatchWithBC(U_bc)
        self.U = CellField(
            mesh,
            ncomp=3,
            ngrow=ngrow,
            name="U",
            fill_patch=fp,
        )
        self.p = CellField(mesh, ncomp=1, ngrow=0, name="p")
        self.phi = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")

        # Per-face pressure BC for the MAC + nodal Poisson solves. Derived from
        # the velocity BC (outflow face → Dirichlet p, inlet/wall → Neumann p);
        # None → the periodic/all-Neumann default preserved below. Stashed on
        # the pressure and phi fields so the free-function solves (dsl.solve,
        # mac_project) can read it.
        if U_bc is not None:
            from .bc import pressure_domain_bc

            self._p_domain_bc = pressure_domain_bc(U_bc, mesh.geom(0))
        else:
            self._p_domain_bc = None
        self.p.pressure_bc = self._p_domain_bc
        self.phi.pressure_bc = self._p_domain_bc

        self._nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)
        self.sol_U = sol_U or {}
        self.sol_p = sol_p or {
            "rtol": 1e-10,
            "atol": 1e-12,
            "maxIter": 200,
            "verbose": 0,
        }

        # Immersed body (API doc §6): geometry lives on ``mesh.body`` (set by
        # the caller, e.g. from meshDict); the method is chosen per field via
        # ``solution["ibm"]`` (``sol_U`` / ``sol_p``). Precompute every
        # distinct method's data eagerly, ready before the first solve.
        ibm_methods = []
        for sol in (self.sol_U, self.sol_p):
            ibm_name = sol.get("ibm")
            if ibm_name is not None:
                method = IBM.lookup(ibm_name)
                if method not in ibm_methods:
                    ibm_methods.append(method)
        if ibm_methods:
            mesh.build_ibm(ibm_methods)

        # UEqn/pEqn are built once — an Equation is a value: terms hold field
        # references (U, p, phi), which survive regrid because fields
        # re-register their MultiFabs. Re-`solve()`d each step().
        self.UEqn = Equation(
            exp.ddt(self.U) + exp.div(self.phi, self.U) - exp.laplacian(self._nu_func, self.U),
            schemes=self.schemes,
        )
        self.pEqn = Equation(
            imp.laplacian(self.dt, self.p) == exp.div(self.U),
            schemes=self.schemes,
        )

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
        mac_project(phi, self.sol_p)

        # 4. Momentum predictor with div-free phi (UEqn built once in __init__,
        #    re-solved each step — terms hold field references, not values).
        self.UEqn.solve(dt=dt, t=t, solution=self.sol_U)

        # 5. Fill BCs on U*
        for lev in range(n_levels):
            U.fill_patch(lev, t)

        # 6. Nodal pressure solve: laplacian(dt, p) = div(U*). sigma tracks the
        #    current (possibly CFL-adaptive) dt on the persistent pEqn term.
        self.pEqn.implicit_lhs.sigma = dt
        self.pEqn.implicit_lhs.coefficient = dt
        self.pEqn.solve(dt=dt, t=t, solution=self.sol_p)

        # 7. Correct U: U^{n+1} = U* - dt * grad(p)
        correct(U, -dt * exp.grad(p))

        # 8. Immersed-body method (API doc §6): per-field solution["ibm"],
        #    e.g. direct forcing pins solid-cell velocity to the wall value so
        #    the body is impermeable and no-slip; the projection on the next
        #    step deflects the flow around the resulting zero-velocity zone.
        #    Applied AFTER the full projection, matching the pre-refactor
        #    apply order exactly (plan 04's acceptance oracle is identical
        #    Cd/Cl/St) — the coupling therefore sits here rather than inside
        #    UEqn.solve()'s explicit-predictor branch, which fires before the
        #    pressure correction and would change the physics.
        ibm_name = self.sol_U.get("ibm")
        if ibm_name is not None:
            method = IBM.lookup(ibm_name)
            data = self.mesh.ibm_data(method)
            method.apply(U, dt, t, data)

        self._t += dt

        # Adaptive time stepping
        if self._cfl is not None:
            max_vel = self._max_velocity()
            if max_vel > 1e-12:
                finest = mesh.n_levels() - 1
                dx_fine = mesh.geom(finest).cell_size()
                self.dt = self._cfl * min(dx_fine) / max_vel

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
            if hasattr(self.p, "_imp_cache"):
                del self.p._imp_cache
            if hasattr(self.phi, "_mac_cache"):
                del self.phi._mac_cache
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
                    (m[1] - 2 * ng) * (m[2] - 2 * ng) * (m[3] - 2 * ng) for m in mf.fab_metadata()
                )
                total_cells += lev_cells
                nboxes = len(mf.fab_metadata())
                layout = blockamr.build_tile_layout(mf, BF)
                print(
                    f"    lev {lev}: {lev_cells:,} cells, {nboxes} boxes, "
                    f"tiles={layout.n_tiles} (padded={layout.n_tiles_padded}), bf={BF}"
                )
            print(f"    total: {total_cells:,} cells")

    def write_plotfile(self, name, fields=None):
        """Write a plotfile. Works for both single-level and AMR."""
        import os
        import shutil

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
                combined = blockamr.MultiFab(mesh.box_array(lev), mesh.dm(lev), total_ncomp, 0)
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
                name,
                mfs[0],
                varnames,
                mesh.geom(0),
                self._t,
                0,
            )
        else:
            blockamr.write_multilevel_plotfile(
                name,
                n_levels,
                mfs,
                varnames,
                [mesh.geom(lev) for lev in range(n_levels)],
                self._t,
                [0] * n_levels,
                [mesh.ref_ratio(lev) for lev in range(n_levels - 1)],
            )

    def _max_velocity(self):
        """Return max |U| across all levels (conservative — includes ghost cells)."""
        max_sq = jnp.float32(0.0)
        for lev in range(self.mesh.n_levels()):
            mf = self.U.mf[lev]
            if mf is None:
                continue
            flat = mf.contiguous_array()
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
