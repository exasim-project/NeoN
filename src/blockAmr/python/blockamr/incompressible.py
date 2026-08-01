# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""DSL incompressible Navier-Stokes projection — data + free functions.

Two-step projection (MAC + nodal) matching IAMReX/incflo:

    1. interpolate(U, phi)
    2. MAC project phi -> div-free face fluxes
    3. ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    4. laplacian(dt, p) == div(U*)
    5. U -= dt * grad(p)

``build_incompressible`` + ``step`` are the standalone driver for the engine
examples/tests. The neofoam framework solver inlines the same worked example in its
``project`` operation instead, resolving state and fields from the Context at run
time; the two share the identical numerics oracle.
"""

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp

import blockamr
from .bc import pressure_domain_bc
from .dsl import exp, imp
from .dsl.equation import Equation
from .field import CellField, FaceField
from .fillpatch import FillPatchWithBC
from .ibm import IBM
from .operators.correct import correct
from .operators.interpolate import interpolate
from .operators.mac_project import mac_project
from .schemes.div_schemes import Upwind
from .schemes.laplacian_schemes import CentralDiffLaplacian
from .schemes.registry import lookup_scheme


@dataclass
class IncompressibleState:
    """Fields + equations + solve settings for one projection solver.

    ``UEqn`` / ``pEqn`` are built once and re-solved each :func:`step`. ``t`` is
    ADVANCED by ``step``; ``dt`` may be pushed in from an outer time loop (e.g. the
    neofoam backend), and ``step`` overwrites it when ``cfl`` is set.
    """

    mesh: Any
    nu: float
    dt: float
    U: CellField
    p: CellField
    phi: FaceField
    UEqn: Equation
    pEqn: Equation
    sol_U: dict[str, Any]
    sol_p: dict[str, Any]
    p_domain_bc: Any = None
    t: float = 0.0
    cfl: Optional[float] = None

    @property
    def time(self) -> float:
        return self.t


def build_incompressible(
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
) -> IncompressibleState:
    """Build the fields + equations for the incompressible projection.

    Works with both single-level ``Mesh`` and multi-level ``AmrMesh``.

    Parameters
    ----------
    mesh, nu, dt
        Mesh, kinematic viscosity, and (initial) time step.
    schemes : dict, optional
        fvSchemes discretisation scheme names, bound to ``UEqn`` / ``pEqn``.
    sol_U, sol_p : dict, optional
        fvSolution ``solvers['U']`` / ``solvers['p']`` blocks (linear solver +
        tolerances + IBM method), passed to ``.solve(solution=...)`` each step.
    U_bc, fill_patch
        Mutually exclusive velocity boundary-condition strategy.
    cfl : float, optional
        Adaptive time-stepping CFL target (used by :func:`step`).
    """
    if U_bc is not None and fill_patch is not None:
        raise ValueError("Specify either U_bc or fill_patch, not both.")
    if U_bc is None and fill_patch is None:
        raise ValueError("One of U_bc or fill_patch must be provided.")

    schemes = schemes or {}

    # ngrow comes from the widest operator stencil, never hardcoded.
    div_stencil_scheme = lookup_scheme(schemes, ["div(phi,U)"], "div", Upwind())
    div_sw = div_stencil_scheme.stencil_width
    lap_sw = CentralDiffLaplacian().stencil_width
    ngrow = max(div_sw, lap_sw)

    fp = fill_patch if fill_patch is not None else FillPatchWithBC(U_bc)
    U = CellField(mesh, ncomp=3, ngrow=ngrow, name="U", fill_patch=fp)
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    phi = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")

    # Per-face pressure BC for the MAC + nodal Poisson solves, stashed on p / phi so
    # the free-function solves (dsl.solve, mac_project) can read it.
    p_domain_bc = pressure_domain_bc(U_bc, mesh.geom(0)) if U_bc is not None else None
    p.pressure_bc = p_domain_bc
    phi.pressure_bc = p_domain_bc

    sol_U = sol_U or {}
    sol_p = sol_p or {"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "verbose": 0}

    # IBM geometry lives on ``mesh.body``; the method is per field via solution["ibm"].
    ibm_methods = []
    for sol in (sol_U, sol_p):
        ibm_name = sol.get("ibm")
        if ibm_name is not None:
            method = IBM.lookup(ibm_name)
            if method not in ibm_methods:
                ibm_methods.append(method)
    if ibm_methods:
        mesh.build_ibm(ibm_methods)

    # Built once: terms hold field references, which survive regrid. ``nu`` must stay a
    # numeric constant, not a lambda — jax collapses a constant callable gamma to the
    # same value, but the cpp backend rejects callables outright.
    UEqn = Equation(
        exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
        schemes=schemes,
    )
    pEqn = Equation(
        imp.laplacian(dt, p) == exp.div(U),
        schemes=schemes,
    )

    return IncompressibleState(
        mesh=mesh,
        nu=nu,
        dt=dt,
        U=U,
        p=p,
        phi=phi,
        UEqn=UEqn,
        pEqn=pEqn,
        sol_U=sol_U,
        sol_p=sol_p,
        p_domain_bc=p_domain_bc,
        cfl=cfl,
    )


def step(state: IncompressibleState) -> None:
    """Advance one time step using the DSL (two-step MAC + nodal projection).

    1. Fill BCs on U
    2. Interpolate U -> phi (not div-free)
    3. MAC projection: make phi div-free
    4. Momentum predictor with div-free phi
    5. Fill BCs on U*
    6. Nodal pressure solve: laplacian(dt, p) = div(U*)
    7. Correct U: U^{n+1} = U* - dt * grad(p)
    8. Immersed-body method (per-field ``solution["ibm"]``), applied AFTER the
       full projection.
    """
    dt = state.dt
    U = state.U
    p = state.p
    phi = state.phi
    t = state.t
    mesh = state.mesh
    n_levels = mesh.n_levels()

    for lev in range(n_levels):
        U.fill_patch(lev, t)

    interpolate(U, phi)
    mac_project(phi, state.sol_p)

    state.UEqn.solve(dt=dt, t=t, solution=state.sol_U)

    for lev in range(n_levels):
        U.fill_patch(lev, t)

    state.pEqn.implicit_lhs.sigma = dt
    state.pEqn.implicit_lhs.coefficient = dt
    state.pEqn.solve(dt=dt, t=t, solution=state.sol_p)

    correct(U, -dt * exp.grad(p))

    ibm_name = state.sol_U.get("ibm")
    if ibm_name is not None:
        method = IBM.lookup(ibm_name)
        method.apply(U, dt, t, mesh.ibm_data(method))

    state.t += dt

    if state.cfl is not None:
        max_vel = max_velocity(U)
        if max_vel > 1e-12:
            finest = mesh.n_levels() - 1
            dx_fine = mesh.geom(finest).cell_size()
            state.dt = state.cfl * min(dx_fine) / max_vel


def regrid_fields(state: IncompressibleState, tag) -> None:
    """Regrid the AMR mesh and invalidate the solver caches. No-op single-level."""
    from .mesh import AmrMesh

    mesh = state.mesh
    if not isinstance(mesh, AmrMesh):
        return

    # Fill ghost cells so tagging stencils have valid data
    for lev in range(mesh.n_levels()):
        state.U.fill_patch(lev, state.t)
    mesh.regrid(state.t, tag=tag)

    # The grids changed, so the cached solver objects are stale.
    if hasattr(state.p, "_imp_cache"):
        del state.p._imp_cache
    if hasattr(state.phi, "_mac_cache"):
        del state.phi._mac_cache

    from .dsl.solve import BF

    total_cells = 0
    print(f"  Regrid: {mesh.n_levels()} levels")
    for lev in range(mesh.n_levels()):
        mf = state.U.mf[lev]
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


def write_plotfile(mesh, t, name, fields) -> None:
    """Write an AMReX plotfile of *fields* at time *t*. Single-level or AMR."""
    import os
    import shutil

    if os.path.exists(name):
        shutil.rmtree(name)

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
            t,
            0,
        )
    else:
        blockamr.write_multilevel_plotfile(
            name,
            n_levels,
            mfs,
            varnames,
            [mesh.geom(lev) for lev in range(n_levels)],
            t,
            [0] * n_levels,
            [mesh.ref_ratio(lev) for lev in range(n_levels - 1)],
        )


def max_velocity(field) -> float:
    """Return max |field| across all levels (conservative — includes ghost cells)."""
    max_sq = jnp.float32(0.0)
    mesh = field.mesh
    for lev in range(mesh.n_levels()):
        mf = field.mf[lev]
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
