# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TDD-2 Cycles 6 & 7: solve with single-level Mesh and multi-level AmrMesh."""

import jax.numpy as jnp
import numpy as np
import pytest

import blockamr
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh, AmrMesh
from blockamr.dsl import exp, imp, solve, Equation
from blockamr.operators.div import Div
from blockamr.schemes.div_schemes import Upwind


def _tag_all(lev, tags, time, ngrow):
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def test_solve_single_level_constant_advection(blockamr_session):
    """Constant phi advected by constant velocity stays constant."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)

    mesh = Mesh(ba, dm, geom)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")

    # Set phi to constant 5.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 5.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    # Set face velocity to constant 1.0 in all directions
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)

    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())
    solve(expr, t=0.0, dt=0.001)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        # Interior cells should remain ~5.0 (constant advected by constant)
        ng = phi.mf[0].n_grow()
        s = slice(ng, -ng if ng else None)
        assert np.allclose(arr[s, s, s, 0], 5.0, atol=1e-10)


def test_solve_multilevel_average_down(blockamr_session):
    """After solve on 2 levels, coarse and fine exist and don't crash."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)

    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=_tag_all)

    # Set constant values on all levels
    for lev in range(mesh.n_levels()):
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            arr = phi.mf[lev].copy_to_host(mfi)
            arr[:] = 5.0
            phi.mf[lev].copy_from(mfi, arr)
        phi.fill_patch(lev, 0.0)
        for d in range(3):
            for mfi in blockamr.MFIterator(ff[lev][d].mf):
                arr = ff[lev][d].mf.copy_to_host(mfi)
                arr[:] = 1.0
                ff[lev][d].mf.copy_from(mfi, arr)

    expr = exp.ddt(phi) + Div(ff, phi, scheme=Upwind())
    solve(expr, t=0.0, dt=0.001)

    # After solve, coarse and fine should exist
    assert phi.mf[0] is not None
    assert phi.mf[1] is not None


# ---------------------------------------------------------------------------
# Plan 02: solve(equation, *, dt, t, solution) — implicit dispatch + solution
# ---------------------------------------------------------------------------


def _make_periodic_mesh(n):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _make_divergent_pressure_system(n=16):
    """A CellField U with non-zero divergence and a pressure field p."""
    mesh, geom = _make_periodic_mesh(n)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")

    pi = np.pi
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(U.mf[0]):
        arr = U.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = np.sin(2 * pi * x)
        U.mf[0].copy_from(mfi, arr)
    U.fill_patch(0, 0.0)
    return U, p


def test_solve_implicit_rejects_snake_case_solution_keys(blockamr_session):
    """A dropped snake_case solution key raises, naming the new spelling."""
    U, p = _make_divergent_pressure_system()
    dt = 0.01
    eqn = imp.laplacian(dt, p) == exp.div(U)
    with pytest.raises(ValueError, match="maxIter"):
        solve(eqn, dt=dt, solution={"max_iter": 200})


def test_solve_implicit_bottom_solver_change_triggers_cache_rebuild(blockamr_session):
    """A changed bottomSolver between two .solve() calls is honoured (rebuild)."""
    U, p = _make_divergent_pressure_system()
    dt = 0.01
    eqn = imp.laplacian(dt, p) == exp.div(U)

    solve(eqn, dt=dt, solution={"bottomSolver": "cg"})
    cache1 = p._imp_cache
    assert cache1.key[-1] == "cg"

    solve(eqn, dt=dt, solution={"bottomSolver": "bicgstab"})
    cache2 = p._imp_cache
    assert cache2.key[-1] == "bicgstab"
    assert cache2 is not cache1, "bottomSolver change did not trigger a cache rebuild"


def test_equation_solve_implicit_end_to_end(blockamr_session):
    """Equation.solve()'s implicit branch runs the MLMG solve end-to-end."""
    U, p = _make_divergent_pressure_system()
    dt = 0.01
    eqn = Equation(imp.laplacian(dt, p) == exp.div(U))

    eqn.solve(dt=dt, solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200})

    assert p._imp_cache.mlmg.get_num_iters() > 0, "MLMG did zero iterations"
    assert p.grad is not None, "p.grad was not set"
    max_grad = max(float(jnp.max(jnp.abs(g))) for lev_grads in p.grad for g in lev_grads)
    assert max_grad > 0, "Gradient is zero"
