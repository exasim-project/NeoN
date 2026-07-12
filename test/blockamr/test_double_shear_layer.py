# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the double shear layer example (single-level and AMR)."""

import numpy as np
import jax.numpy as jnp

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh, AmrMesh
from neon.blockamr.field import CellField
from neon.blockamr.incompressible import build_incompressible, max_velocity, regrid_fields, step
from neon.blockamr.fillpatch import FillPatchCellConservative
from neon.blockamr.operators.interpolate import interpolate
from neon.blockamr.schemes.div_schemes import VanLeer


def _shear_layer_ic(mf, geom, rho=30.0, delta=0.05):
    """Set double shear layer initial condition on a MultiFab."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1

        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])

        x2d = xs[:, None] * jnp.ones((1, ny))
        y2d = jnp.ones((nx, 1)) * ys[None, :]

        u = jnp.where(
            y2d <= 0.5,
            jnp.tanh(rho * (y2d - 0.25)),
            jnp.tanh(rho * (0.75 - y2d)),
        )
        v = delta * jnp.sin(2.0 * jnp.pi * x2d)
        w = jnp.zeros_like(u)

        vals = jnp.stack([u, v, w], axis=-1)
        vals = vals[:, :, None, :] * jnp.ones((1, 1, nz, 1))
        mf.copy_from(mfi, vals)


def _make_single_level_solver(N=16, Re=1000, cfl=0.25, max_size=64):
    """Create a single-level double shear layer solver."""
    nu = 1.0 / Re
    dt = cfl / N
    Nz = 4

    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    solver = build_incompressible(
        mesh,
        nu,
        dt,
        fill_patch=FillPatchCellConservative(),
        sol_p={"rtol": 1e-10, "atol": 1e-8, "maxIter": 200, "verbose": 0},
    )
    _shear_layer_ic(solver.U.mf[0], geom)
    return solver, geom


def _measure_face_div(phi, mesh):
    """Compute face flux divergence statistics.

    Returns (max_abs, total_sum): max|div(phi)| and sum(div(phi)) over domain.
    """
    dx = mesh.geom(0).cell_size()
    max_abs = 0.0
    total_sum = 0.0
    face_arrs = [phi[0][d].mf.arrays() for d in range(3)]
    n_boxes = len(face_arrs[0])
    for bi in range(n_boxes):
        div_val = None
        for d in range(3):
            f = face_arrs[d][bi][:, :, :, 0]
            ng = phi[0][d].mf.n_grow()
            nc = [int(f.shape[ax]) - 2 * ng - (1 if ax == d else 0) for ax in range(3)]
            sl_hi = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_lo = [slice(ng, ng + nc[ax]) for ax in range(3)]
            sl_hi[d] = slice(ng + 1, ng + 1 + nc[d])
            sl_lo[d] = slice(ng, ng + nc[d])
            contrib = (f[tuple(sl_hi)] - f[tuple(sl_lo)]) / dx[d]
            div_val = contrib if div_val is None else div_val + contrib
        max_abs = max(max_abs, float(jnp.max(jnp.abs(div_val))))
        total_sum += float(jnp.sum(div_val))
    return max_abs, total_sum


def _make_example_solver(N=128, Re=10000, cfl=0.25, max_size=64, rho=80.0, delta=0.05):
    """Create solver matching the double_shear_layer.py example parameters."""
    nu = 1.0 / Re
    dt = cfl / N
    Nz = 4

    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    solver = build_incompressible(
        mesh,
        nu,
        dt,
        fill_patch=FillPatchCellConservative(),
        sol_p={"rtol": 0, "atol": 1e-8, "maxIter": 200, "verbose": 0},
        schemes={"div(phi,U)": VanLeer()},
        cfl=cfl,
    )
    _shear_layer_ic(solver.U.mf[0], geom, rho=rho, delta=delta)
    return solver, mesh


def _tag_all(lev, tags, time, ngrow):
    """Tag every cell for refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def _make_amr_solver(N=16, Re=1000, cfl=0.25, max_level=1, max_size=32):
    """Create a multi-level AMR double shear layer solver."""
    nu = 1.0 / Re
    dt = cfl / N
    Nz = 4

    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = max_level
    for lev in range(max_level):
        info.set_ref_ratio(lev, 2)
    info.set_max_grid_size(0, max_size)
    info.set_blocking_factor(0, 4)
    mesh = AmrMesh(geom, info)

    solver = build_incompressible(
        mesh,
        nu,
        dt,
        fill_patch=FillPatchCellConservative(),
        sol_p={"rtol": 1e-10, "atol": 1e-8, "maxIter": 200, "verbose": 0},
    )

    mesh.init_from_scratch(0.0)
    _shear_layer_ic(solver.U.mf[0], mesh.geom(0))

    # Regrid to create fine levels (tag all cells for a uniform fine level)
    for _ in range(max_level + 1):
        mesh.regrid(0.0, tag=_tag_all)
        for lev in range(mesh.n_levels()):
            _shear_layer_ic(solver.U.mf[lev], mesh.geom(lev))

    return solver, mesh


# --- Single-level tests ---


def test_single_level_velocity_bounded(blockamr_session):
    """Single-level shear layer velocity should remain bounded."""
    solver, geom = _make_single_level_solver(N=16, Re=1000)

    for _ in range(20):
        step(solver)

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(jnp.max(jnp.abs(U_arrs)))
    assert max_vel < 5.0, f"Max velocity {max_vel} — solver may be unstable"
    assert max_vel > 0.01, f"Max velocity {max_vel} — solver may not be running"


def test_single_level_multi_box(blockamr_session):
    """Single-level with multiple boxes (max_size < ncell) should work."""
    solver, geom = _make_single_level_solver(N=32, Re=1000, max_size=16)

    for _ in range(10):
        step(solver)

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(jnp.max(jnp.abs(U_arrs)))
    assert max_vel < 5.0
    assert max_vel > 0.01


# --- AMR tests ---


def test_amr_solver_runs(blockamr_session):
    """AMR solver should run without errors."""
    solver, mesh = _make_amr_solver(N=16, Re=1000, max_level=1)
    assert mesh.n_levels() == 2

    for _ in range(5):
        step(solver)

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(jnp.max(jnp.abs(U_arrs)))
    assert max_vel < 5.0
    assert max_vel > 0.01


def test_amr_fine_level_has_data(blockamr_session):
    """After AMR solve, fine level should have valid data."""
    solver, mesh = _make_amr_solver(N=16, Re=1000, max_level=1)

    for _ in range(5):
        step(solver)

    # Fine level should exist and have data
    assert solver.U.mf[1] is not None
    U1_arrs = solver.U.mf[1].arrays()[0]
    max_vel_fine = float(jnp.max(jnp.abs(U1_arrs)))
    assert max_vel_fine > 0.01, "Fine level velocity is zero — data not propagated"


def test_amr_levels_persist_after_regrid(blockamr_session):
    """AMR levels must survive regrid during the time loop.

    Uses tag_all so the test is independent of the tagging threshold.
    Verifies that solver.regrid() preserves fine levels and that the
    solution remains valid (not NaN, velocity bounded) afterward.
    """
    solver, mesh = _make_amr_solver(N=16, Re=1000, max_level=1)
    assert mesh.n_levels() == 2, "Setup should create 2 levels"

    # Run a few steps, then regrid, then run more steps
    for _ in range(5):
        step(solver)

    regrid_fields(solver, _tag_all)
    assert mesh.n_levels() == 2, f"Fine level lost after regrid: n_levels={mesh.n_levels()}"

    for _ in range(5):
        step(solver)

    # Velocity must be finite and bounded on both levels
    for lev in range(mesh.n_levels()):
        mf = solver.U.mf[lev]
        assert mf is not None, f"Level {lev} MultiFab is None after regrid+step"
        for arr in mf.arrays():
            assert bool(jnp.all(jnp.isfinite(arr))), (
                f"NaN/Inf in velocity on level {lev} after regrid"
            )
        ng = mf.n_grow()
        for arr in mf.arrays():
            u_int = arr[ng:-ng, ng:-ng, ng:-ng, :]
            mag = float(jnp.max(jnp.sqrt(jnp.sum(u_int**2, axis=-1))))
            assert mag < 5.0, f"Velocity blowup on level {lev}: max|U|={mag}"
            assert mag > 0.01, f"Velocity collapsed on level {lev}: max|U|={mag}"


# --- Diagnostic: face flux divergence ---


def test_face_flux_divergence_bounded(blockamr_session):
    """Face flux divergence should stay bounded over 200 steps at N=128.

    Matches example/blockamr/double_shear_layer.py parameters:
    N=128, Re=10000, VanLeer, CFL=0.25, rho=80.

    If the face fluxes are not divergence-free, div(phi) grows over time
    and eventually causes checkerboard instability.
    """
    solver, mesh = _make_example_solver(N=128, Re=10000, cfl=0.25)
    n_steps = 200

    max_div_history = []
    sum_div_history = []
    vel_history = []
    for _ in range(1, n_steps + 1):
        step(solver)

        # Measure div(phi) AFTER the step — phi has been corrected at step 7
        max_div, sum_div = _measure_face_div(solver.phi, mesh)
        max_vel = max_velocity(solver.U)

        max_div_history.append(max_div)
        sum_div_history.append(sum_div)
        vel_history.append(max_vel)

    print("\n  Face flux divergence diagnostic (N=128, Re=10000, VanLeer):")
    print(f"  {'step':>6s}  {'max|div(phi)|':>14s}  {'sum(div(phi))':>14s}  {'max|U|':>10s}")
    for i in [0, 49, 99, 199]:
        print(
            f"  {i + 1:6d}  {max_div_history[i]:14.6e}  {sum_div_history[i]:14.6e}  {vel_history[i]:10.6f}"
        )

    # Velocity must remain bounded
    assert vel_history[-1] < 5.0, f"Velocity blowup: max|U|={vel_history[-1]:.4f} at step {n_steps}"
    assert all(np.isfinite(v) for v in vel_history), "NaN/Inf in velocity history"

    # div(phi) should not grow unboundedly
    assert max_div_history[-1] < 10.0, (
        f"Face flux divergence too large: max|div(phi)|={max_div_history[-1]:.6e} at step {n_steps}"
    )


def test_momentum_solve_max_size_independence(blockamr_session):
    """Momentum solve with interpolated face velocity must not depend on max_size."""
    from neon.blockamr.dsl.solve import solve
    from neon.blockamr.dsl import exp
    from neon.blockamr.field import FaceField

    N, Nz = 32, 4

    def run_momentum(max_size):
        box = blockamr.Box([0, 0, 0], [N - 1, N - 1, Nz - 1])
        rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N])
        geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
        ba = blockamr.BoxArray(box)
        ba.max_size(max_size)
        dm = blockamr.DistributionMapping(ba)
        mesh = Mesh(ba, dm, geom)

        U = CellField(mesh, ncomp=3, ngrow=1, name="U", fill_patch=FillPatchCellConservative())
        _shear_layer_ic(U.mf[0], geom)
        U.fill_patch(0, 0.0)

        phi = FaceField(mesh, ncomp=1, ngrow=1)
        interpolate(U, phi)

        dt = 0.25 / N
        nu = 1e-3
        solve(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U), t=0.0, dt=dt)

        ng = U.mf[0].n_grow()
        all_valid = []
        for arr in U.mf[0].arrays():
            all_valid.append(np.array(arr[ng:-ng, ng:-ng, ng:-ng, :]).ravel())
        return np.sort(np.concatenate(all_valid))

    single_box = run_momentum(max_size=N)
    multi_box = run_momentum(max_size=8)

    np.testing.assert_allclose(
        single_box,
        multi_box,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Momentum solve results depend on max_size (box decomposition)",
    )


def test_full_step_max_size_independence(blockamr_session):
    """Full step(solver) must produce identical valid cells regardless of max_size."""
    N, Nz, Re = 32, 4, 1000

    def run_step(max_size):
        solver, geom = _make_single_level_solver(N=N, Re=Re, max_size=max_size)
        _shear_layer_ic(solver.U.mf[0], geom)
        step(solver)
        mf = solver.U.mf[0]
        ng = mf.n_grow()
        all_valid = []
        for arr in mf.arrays():
            all_valid.append(np.array(arr[ng:-ng, ng:-ng, ng:-ng, :]).ravel())
        return np.sort(np.concatenate(all_valid))

    single = run_step(max_size=N)
    multi = run_step(max_size=8)

    np.testing.assert_allclose(
        single,
        multi,
        rtol=1e-5,
        atol=1e-10,
        err_msg="Full solver step depends on max_size (box decomposition)",
    )
