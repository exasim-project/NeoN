# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""DSL-based incompressible Navier-Stokes solver.

Uses the OpenFOAM-style DSL syntax:

    ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    laplacian(dt, p) == div(U*)
    U -= dt * grad(p)
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

    @property
    def time(self):
        return self._t

    def step(self):
        """Advance one time step using the DSL."""
        dt = self.dt
        U = self.U
        p = self.p
        phi = self.phi
        t = self._t
        mesh = self.mesh
        n_levels = mesh.n_levels()

        # Fill BCs before face interpolation (all levels)
        for lev in range(n_levels):
            U.fill_patch(lev, t)

        # Face flux from cell velocity (all levels)
        interpolate(U, phi)

        # Momentum predictor (explicit, handles all levels internally):
        #   ddt(U) + div(phi, U) - laplacian(nu, U) = 0
        solve(
            exp.ddt(U) + exp.div(phi, U, scheme=self._div_scheme)
            - exp.laplacian(self._nu_func, U),
            t, dt,
        )

        # Fill BCs on U* before pressure solve (all levels)
        for lev in range(n_levels):
            U.fill_patch(lev, t)

        # Pressure correction (implicit nodal Poisson, handles all levels):
        #   laplacian(dt, p) = div(U*)
        solve(imp.laplacian(dt, p) == exp.div(U), schemes=self._schemes_p)

        # Velocity correction: U -= dt * grad(p) (handles all levels)
        correct(U, -dt * exp.grad(p))

        self._t += dt

        # Adaptive time stepping: recompute dt from current max velocity
        # Use the finest level's cell size for the CFL constraint
        if self._cfl is not None:
            max_vel = self._max_velocity()
            if max_vel > 1e-12:
                finest = mesh.n_levels() - 1
                dx_fine = mesh.geom(finest).cell_size()
                self.dt = self._cfl * min(dx_fine) / max_vel

    def regrid(self, tag):
        """Regrid the AMR mesh. No-op for single-level meshes."""
        from .mesh import AmrMesh
        if isinstance(self.mesh, AmrMesh):
            self.mesh.regrid(self._t, tag=tag)
            # Invalidate pressure solver cache — grids changed
            if hasattr(self.p, '_imp_solver'):
                del self.p._imp_solver
            mesh = self.mesh
            parts = []
            for lev in range(mesh.n_levels()):
                dom = mesh.geom(lev).domain()
                lo = dom.small_end()
                hi = dom.big_end()
                ncells = 1
                for d in range(3):
                    ncells *= (hi[d] - lo[d] + 1)
                nboxes = len(self.U.mf[lev].arrays())
                parts.append(f"lev{lev}: {ncells} cells, {nboxes} boxes")
            print(f"  Regrid: {', '.join(parts)}")

    def write_plotfile(self, name, fields=None):
        """Write a plotfile. Works for both single-level and AMR.

        Parameters
        ----------
        name : str
            Plotfile directory name.
        fields : list[CellField], optional
            Fields to write. Defaults to [self.U].
            Variable names are derived from each field's name attribute.
        """
        import os, shutil
        if os.path.exists(name):
            shutil.rmtree(name)

        if fields is None:
            fields = [self.U]

        mesh = self.mesh
        n_levels = mesh.n_levels()

        # Build variable names from fields
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
            # Combine multiple fields into one MultiFab per level
            total_ncomp = sum(f.ncomp for f in fields)
            mfs = []
            for lev in range(n_levels):
                combined = blockamr.MultiFab(
                    mesh.box_array(lev), mesh.dm(lev), total_ncomp, 0)
                # Stack valid-region arrays from each field
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
        """Return max |U| across all levels (valid cells only)."""
        max_val = 0.0
        for lev in range(self.mesh.n_levels()):
            mf = self.U.mf[lev]
            if mf is None:
                continue
            ng = mf.n_grow()
            for arr in mf.arrays():
                u_int = arr[ng:-ng, ng:-ng, ng:-ng, :]
                mag = jnp.sqrt(jnp.sum(u_int ** 2, axis=-1))
                max_val = max(max_val, float(jnp.max(mag)))
        return max_val
