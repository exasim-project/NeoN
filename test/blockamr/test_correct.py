# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the correct operator: U -= dt * grad(p)."""

import math

import jax.numpy as jnp
import numpy as np

import blockamr
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh
from blockamr.dsl import exp, imp, solve
from blockamr.operators.interpolate import interpolate
from blockamr.operators.correct import correct


def _make_mesh(N, max_size=None):
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _set_cellfield(field, geom, func):
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(field.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = func(X, Y, Z)
        field.mf[0].copy_from(mfi, vals)
    field.fill_patch(0, 0.0)


def test_correct_divergence_free(blockamr_session):
    """After pressure projection, velocity is divergence-free.

    U* = sin(2π·y) x-hat → has div but project removes it.
    """
    N = 16
    mesh, geom = _make_mesh(N)
    pi = math.pi
    dt = 0.001
    nu = 0.01

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    def shear_vel(X, Y, Z):
        return jnp.stack([jnp.sin(2 * pi * Y), jnp.zeros_like(X), jnp.zeros_like(X)], axis=-1)

    _set_cellfield(U, geom, shear_vel)

    # One explicit step → U* with non-zero divergence
    interpolate(U, phi)
    nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)
    solve(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu_func, U), t=0.0, dt=dt)
    U.fill_patch(0, 0.0)

    # Pressure projection
    solve(
        imp.laplacian(dt, p) == exp.div(U),
        solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "verbose": 0},
    )
    correct(U, -dt * exp.grad(p))
    U.fill_patch(0, 0.0)

    # Check divergence via compDivergence
    ba = mesh.box_array(0)
    dm = mesh.dm(0)
    vel3 = blockamr.MultiFab(ba, dm, 3, 1)
    grown = U.mf[0].grown_arrays()[0]
    for mfi in blockamr.MFIterator(vel3):
        vel3.copy_grown_from(mfi, np.asfortranarray(np.array(grown)))

    dom = geom.domain()
    lo = dom.small_end()
    hi = dom.big_end()
    nodal_box = blockamr.Box(lo, [hi[0] + 1, hi[1] + 1, hi[2] + 1])
    nodal_ba = blockamr.BoxArray(nodal_box)
    nodal_ba.max_size(N + 1)
    rhs = blockamr.MultiFab(nodal_ba, dm, 1, 0)

    lp = blockamr.MLNodeLaplacian(geom, ba, dm, blockamr.LPInfo(), dt)
    is_per = geom.is_periodic()
    lo_bc = [
        blockamr.LinOpBCType.Periodic if is_per[d] else blockamr.LinOpBCType.Neumann
        for d in range(3)
    ]
    lp.set_domain_bc(lo_bc, lo_bc[:])
    lp.comp_divergence(rhs, vel3)

    max_div = float(jnp.max(jnp.abs(rhs.arrays()[0])))
    print(f"\nmax|div(U)| after correction = {max_div:.2e}")
    assert max_div < 1e-8, f"max|div(U)| = {max_div:.2e} — not divergence-free"


def test_correct_zero_pressure_no_change(blockamr_session):
    """Correction with zero pressure gradient leaves U unchanged.

    Uses the full pressure solve path with zero RHS so grad(p) = 0.
    """
    N = 16
    mesh, geom = _make_mesh(N)
    dt = 0.001

    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")

    # Set uniform velocity (div-free)
    def const_vel(X, Y, Z):
        return jnp.stack([jnp.ones_like(X), jnp.zeros_like(X), jnp.zeros_like(X)], axis=-1)

    _set_cellfield(U, geom, const_vel)

    # Record U before
    u_before = np.array(U.mf[0].arrays()[0][:, :, :, 0])

    # Pressure solve with div-free U → p ≈ 0 → grad(p) ≈ 0
    solve(
        imp.laplacian(dt, p) == exp.div(U),
        solution={"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "verbose": 0},
    )
    correct(U, -dt * exp.grad(p))

    u_after = np.array(U.mf[0].arrays()[0][:, :, :, 0])
    ng = U.mf[0].n_grow()
    # Compare valid region only
    diff = np.abs(u_after[ng:-ng, ng:-ng, ng:-ng] - u_before[ng:-ng, ng:-ng, ng:-ng])
    max_diff = float(np.max(diff))
    print(f"\nCorrect with div-free U: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-8, f"U changed with div-free field: max_diff={max_diff}"
