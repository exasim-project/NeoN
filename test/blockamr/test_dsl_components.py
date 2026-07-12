# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Unit tests for DSL solver components with analytical solutions."""

import math

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.dsl import exp, imp, solve
from neon.blockamr.bc import (
    BoundaryCondition,
    DirichletBC,
    VectorBC,
    fixedValue,
    noSlip,
)
from neon.blockamr.fillpatch import FillPatchWithBC
from neon.blockamr.operators.interpolate import interpolate
from neon.blockamr.operators.correct import correct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_periodic_mesh(N, max_size=None):
    """Fully periodic mesh on [0,1]^3."""
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _make_cavity_mesh(N, max_size=None):
    """Non-periodic mesh on [0,1]^3, periodic in z only."""
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _set_cellfield_from_func(field, geom, func):
    """Set a CellField from func(X, Y, Z) → array of shape (nx, ny, nz) or (nx, ny, nz, ncomp)."""
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(field.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = func(X, Y, Z)
        field.mf[0].copy_from(mfi, vals)
    field.fill_patch(0, 0.0)


def _set_face_field_constant(face_field, geom, ux, uy, uz):
    """Set all face fluxes to constant values."""
    vals = [ux, uy, uz]
    for d in range(3):
        face_mf = face_field[0][d].mf
        for mfi in blockamr.MFIterator(face_mf):
            arr = face_mf.copy_to_host(mfi)
            arr[:] = vals[d]
            face_mf.copy_from(mfi, arr)


# ---------------------------------------------------------------------------
# Test 1: Interpolation of linear field is exact
# ---------------------------------------------------------------------------


def test_interpolate_linear_field_exact(blockamr_session):
    """Linear interpolation of a linear field U_x = x is exact at face centres.

    For a linear field, face[i+1/2] = 0.5*(cell[i] + cell[i+1]) = (i+1)*dx,
    which equals the face position exactly. Check interior faces (1..N-1).
    """
    N = 16
    mesh, geom = _make_periodic_mesh(N)
    dx_val = 1.0 / N

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    # U_x = x (linear), U_y = U_z = 0
    def linear_x(X, Y, Z):
        return jnp.stack([X, jnp.zeros_like(X), jnp.zeros_like(X)], axis=-1)

    _set_cellfield_from_func(U, geom, linear_x)
    interpolate(U, phi)

    # Read cell values with ghosts to compute expected face values
    cell_grown = np.array(U.mf[0].grown_arrays()[0][:, :, :, 0])  # (N+2, N+2, N+2)
    cell_ng = U.ngrow

    # Read x-face valid data
    face_mf = phi[0][0].mf
    for mfi in blockamr.MFIterator(face_mf):
        face_valid = np.array(face_mf.copy_to_host(mfi)[:, :, :, 0])  # (N+1, N, N)
        break

    # Interior faces (i=1..N-1): face[i] = 0.5*(cell[i-1] + cell[i])
    # These don't touch boundary/ghost wrapping
    jy, jz = N // 2, N // 2
    for i in range(1, N):
        cell_left = cell_grown[cell_ng + i - 1, cell_ng + jy, cell_ng + jz]
        cell_right = cell_grown[cell_ng + i, cell_ng + jy, cell_ng + jz]
        expected = 0.5 * (cell_left + cell_right)
        got = face_valid[i, jy, jz]
        assert abs(got - expected) < 1e-12, f"Face {i}: got {got:.6f}, expected {expected:.6f}"


# ---------------------------------------------------------------------------
# Test 2: Advection-diffusion convergence with fixedValue BCs
# ---------------------------------------------------------------------------


def _advdiff_error(N, nu, n_steps=1):
    """Compute error of advection-diffusion of sin(2π·x) in uniform flow U=1.

    Uses fixedValue BCs (Dirichlet) with values from the analytical solution.
    Returns max error after n_steps Forward Euler steps.
    """
    pi = math.pi
    k = 2 * pi
    dx_val = 1.0 / N
    # dt = O(dx²) so temporal error is subdominant
    dt = 0.5 * dx_val**2 / nu

    mesh, geom = _make_cavity_mesh(N)
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()

    # Analytical solution: φ(x,t) = sin(k*(x - t)) * exp(-ν*k²*t)
    def exact(x, t):
        return jnp.sin(k * (x - t)) * jnp.exp(-nu * k**2 * t)

    # BCs: Dirichlet with value at the boundary (value = average of exact at face)
    # For simplicity use the exact cell-centre value at the boundary cells
    bc_lo_val = float(exact(0.5 * dx_val, 0.0))
    bc_hi_val = float(exact(1.0 - 0.5 * dx_val, 0.0))
    scalar_bc = BoundaryCondition(
        lo=[DirichletBC(bc_lo_val), DirichletBC(bc_lo_val), DirichletBC(0.0)],
        hi=[DirichletBC(bc_hi_val), DirichletBC(bc_hi_val), DirichletBC(0.0)],
    )

    phi_field = CellField(
        mesh,
        ncomp=1,
        ngrow=1,
        name="phi",
        fill_patch=FillPatchWithBC(scalar_bc),
    )

    # Initial condition: φ(x,0) = sin(2π·x)
    def init_func(X, Y, Z):
        return jnp.sin(k * X)

    _set_cellfield_from_func(phi_field, geom, init_func)

    # Face flux: constant U_adv = 1 in x
    face_flux = FaceField(mesh, ncomp=1, ngrow=1, name="ff")
    _set_face_field_constant(face_flux, geom, 1.0, 0.0, 0.0)

    nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)

    # Time-step
    t = 0.0
    for _ in range(n_steps):
        solve(
            exp.ddt(phi_field) + exp.div(face_flux, phi_field) - exp.laplacian(nu_func, phi_field),
            t=t,
            dt=dt,
        )
        t += dt

    # Compute error vs analytical at t
    max_err = 0.0
    for mfi in blockamr.MFIterator(phi_field.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        exact_vals = exact(xs, t)
        # Take middle y, z slice
        arr = phi_field.mf[0].copy_to_host(mfi)
        numerical = jnp.array(arr[:, ny // 2, nz // 2, 0])
        err = float(jnp.max(jnp.abs(numerical - exact_vals)))
        max_err = max(max_err, err)
    return max_err


def test_advection_diffusion_convergence(blockamr_session):
    """Advection-diffusion of sin(2π·x) in uniform flow converges at O(dx).

    Uses fixedValue BCs. Tests both exp.div and exp.laplacian together.
    First-order upwind limits overall convergence to O(dx).
    """
    nu = 0.01
    errors = []
    for N in [16, 32, 64]:
        err = _advdiff_error(N, nu, n_steps=1)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 1.5, f"Ratio 16→32: {ratio_1:.2f}, expected ~2"
    assert ratio_2 > 1.5, f"Ratio 32→64: {ratio_2:.2f}, expected ~2"


# ---------------------------------------------------------------------------
# Test 3: Pressure solve with known divergence
# ---------------------------------------------------------------------------


def test_pressure_solve_known_divergence(blockamr_session):
    """Implicit pressure solve produces non-zero solution for divergent velocity.

    U* = (sin(2π·x), 0, 0) has div(U*) = 2π·cos(2π·x) ≠ 0.
    The MLMG solver should run >0 iterations.
    """
    N = 16
    mesh, geom = _make_periodic_mesh(N)
    dt = 0.01
    pi = math.pi

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")

    def sin_vel(X, Y, Z):
        return jnp.stack(
            [
                jnp.sin(2 * pi * X),
                jnp.zeros_like(X),
                jnp.zeros_like(X),
            ],
            axis=-1,
        )

    _set_cellfield_from_func(U, geom, sin_vel)

    solve(
        imp.laplacian(dt, p) == exp.div(U),
        solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "verbose": 0},
    )

    s = p._imp_cache
    assert s.mlmg.get_num_iters() > 0, "MLMG did zero iterations — RHS may be zero"
    assert s.mlmg.get_init_residual() > 0, "Initial residual is zero"
    assert p.grad is not None, "p.grad was not set"
    max_grad = max(float(jnp.max(jnp.abs(g))) for lev_grads in p.grad for g in lev_grads)
    assert max_grad > 0, "Gradient is zero"


# ---------------------------------------------------------------------------
# Test 4: Pressure correction yields divergence-free velocity
# ---------------------------------------------------------------------------


def test_pressure_correction_divergence_free(blockamr_session):
    """After pressure projection, velocity is divergence-free.

    Start with U_x = sin(2π·y), do one explicit step to create non-zero
    divergence, then project. Check div(U_corrected) ≈ 0.
    """
    N = 16
    mesh, geom = _make_periodic_mesh(N)
    nu = 0.01
    dt = 0.001
    pi = math.pi

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    def shear_vel(X, Y, Z):
        return jnp.stack(
            [
                jnp.sin(2 * pi * Y),
                jnp.zeros_like(X),
                jnp.zeros_like(X),
            ],
            axis=-1,
        )

    _set_cellfield_from_func(U, geom, shear_vel)

    nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)

    # One explicit step → U* may have non-zero divergence
    interpolate(U, phi)
    solve(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu_func, U), t=0.0, dt=dt)
    U.fill_patch(0, 0.0)

    # Pressure projection
    solve(
        imp.laplacian(dt, p) == exp.div(U),
        solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "verbose": 0},
    )
    correct(U, -dt * exp.grad(p))
    U.fill_patch(0, 0.0)

    # Verify divergence-free by recomputing nodal divergence via compDivergence
    ba = mesh.box_array(0)
    dm = mesh.dm(0)

    vel3 = blockamr.MultiFab(ba, dm, 3, 1)
    grown = U.mf[0].grown_arrays()[0]
    grown_np = np.asfortranarray(np.array(grown))
    for mfi in blockamr.MFIterator(vel3):
        vel3.copy_grown_from(mfi, grown_np)

    dom = geom.domain()
    lo = dom.small_end()
    hi = dom.big_end()
    nodal_box = blockamr.Box(lo, [hi[0] + 1, hi[1] + 1, hi[2] + 1])
    nodal_ba = blockamr.BoxArray(nodal_box)
    nodal_ba.max_size(N + 1)
    rhs = blockamr.MultiFab(nodal_ba, dm, 1, 0)

    sigma = dt
    lp = blockamr.MLNodeLaplacian(geom, ba, dm, blockamr.LPInfo(), sigma)
    is_per = geom.is_periodic()
    lo_bc = [
        blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
        for d in range(3)
    ]
    lp.set_domain_bc(lo_bc, lo_bc[:])
    lp.comp_divergence(rhs, vel3)

    rhs_arr = rhs.arrays()[0]
    max_div = float(jnp.max(jnp.abs(rhs_arr)))
    assert max_div < 1e-8, f"max|div(U)| = {max_div:.2e} after projection — should be ~0"


# ---------------------------------------------------------------------------
# Test 5: DSL solver lid-driven cavity physical sanity
# ---------------------------------------------------------------------------


def test_dsl_solver_lid_cavity_physical(blockamr_session):
    """DSL solver produces physically reasonable lid-driven cavity flow.

    After 500 steps at Re=100, velocity should be bounded, non-zero,
    and show lid-driven flow structure.
    """
    from neon.blockamr.incompressible import build_incompressible, step

    N = 16
    Re = 100
    nu = 1.0 / Re
    cfl = 0.25
    dt = cfl / N

    mesh, geom = _make_cavity_mesh(N)
    U_bc = VectorBC(
        xlo=noSlip(),
        xhi=noSlip(),
        ylo=noSlip(),
        yhi=fixedValue([1, 0, 0]),
    )
    solver = build_incompressible(mesh, nu, dt, U_bc=U_bc)

    for _ in range(500):
        step(solver)

    # Extract velocity data
    ng = solver.U.mf[0].n_grow()
    U_arrs = np.array(solver.U.mf[0].arrays()[0])

    # Valid region
    u_valid = U_arrs[ng : ng + N, ng : ng + N, ng : ng + N, :]

    max_vel = np.max(np.abs(u_valid))
    assert max_vel < 2.0, f"Max velocity {max_vel:.3f} — solver may be unstable"
    assert max_vel > 0.01, f"Max velocity {max_vel:.3f} — solver may not be running"

    # Near lid (top row of cells): u_x should be significant
    u_top = np.mean(np.abs(u_valid[:, -1, :, 0]))
    assert u_top > 0.3, f"u_x near lid = {u_top:.3f} — lid not driving flow"

    # Near bottom wall: u_x should be small
    u_bottom = np.mean(np.abs(u_valid[:, 0, :, 0]))
    assert u_bottom < 0.3, f"u_x near bottom = {u_bottom:.3f} — no-slip not working"
