# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the double shear layer example (single-level and AMR)."""

import numpy as np
import jax.numpy as jnp

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh, AmrMesh
from neon.blockamr.field import CellField
from neon.blockamr.dsl_solver import DSLIncompressibleSolver
from neon.blockamr.fillpatch import FillPatchCellConservative


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

    solver = DSLIncompressibleSolver(
        mesh, nu, dt, fill_patch=FillPatchCellConservative(),
        schemes_p={"rtol": 1e-10, "atol": 1e-8, "max_iter": 200, "verbose": 0},
    )
    _shear_layer_ic(solver.U.mf[0], geom)
    return solver, geom


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

    solver = DSLIncompressibleSolver(
        mesh, nu, dt, fill_patch=FillPatchCellConservative(),
        schemes_p={"rtol": 1e-10, "atol": 1e-8, "max_iter": 200, "verbose": 0},
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
        solver.step()

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(jnp.max(jnp.abs(U_arrs)))
    assert max_vel < 5.0, f"Max velocity {max_vel} — solver may be unstable"
    assert max_vel > 0.01, f"Max velocity {max_vel} — solver may not be running"


def test_single_level_multi_box(blockamr_session):
    """Single-level with multiple boxes (max_size < ncell) should work."""
    solver, geom = _make_single_level_solver(N=32, Re=1000, max_size=16)

    for _ in range(10):
        solver.step()

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
        solver.step()

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(jnp.max(jnp.abs(U_arrs)))
    assert max_vel < 5.0
    assert max_vel > 0.01


def test_amr_fine_level_has_data(blockamr_session):
    """After AMR solve, fine level should have valid data."""
    solver, mesh = _make_amr_solver(N=16, Re=1000, max_level=1)

    for _ in range(5):
        solver.step()

    # Fine level should exist and have data
    assert solver.U.mf[1] is not None
    U1_arrs = solver.U.mf[1].arrays()[0]
    max_vel_fine = float(jnp.max(jnp.abs(U1_arrs)))
    assert max_vel_fine > 0.01, "Fine level velocity is zero — data not propagated"
