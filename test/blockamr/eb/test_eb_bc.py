# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Stage B — Boundary-condition fill correctness on EB and non-EB meshes.

These tests pin down the behaviour of ``FillPatchWithBC`` so that any
future change to BC handling that breaks the inflow/outflow flow-past-
cylinder example trips a fast, narrow test failure rather than a
mysterious solver blow-up.

Convention (verified empirically against ``mf.fill_domain_boundary``):

  - **DirichletBC** on ``U_x`` with wall value ``v`` is a linear
    extrapolation through the face:
        ghost = 2 * v − interior
    so a uniform interior of 0.5 with v=1.0 yields a ghost of 1.5.
  - **NeumannBC** is zero-gradient: ghost = interior.

The hot test for the current Re=20 blow-up is **B4** — does one
``solver.step()`` preserve the inflow Dirichlet ghost? If B4 fails on
a *non-EB* mesh, the bug is in the predictor or the corrector
(independent of EB), and the search collapses to those two helpers.
"""

import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import jax.numpy as jnp
import numpy as np
import pytest

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh
from neon.blockamr.field import CellField
from neon.blockamr.fillpatch import FillPatchWithBC
from neon.blockamr.bc import VectorBC, fixedValue, NeumannBC, slipWall
from neon.blockamr.dsl_solver import DSLIncompressibleSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_box_mesh(nx=16, ny=8, nz=4, Lx=2.0, Ly=1.0, periodic_z=True,
                     eb_factory=None):
    """A single-box mesh with non-periodic x and y, periodic z.

    Single-box layout (no max_size split) means there is exactly one
    fab whose ghost cells touch every domain face — assertions can
    index ghost slabs directly with ``arr[0, :, :, c]`` (xlo) etc.
    """
    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [Lx, Ly, Ly * nz / ny])
    is_per = [0, 0, 1 if periodic_z else 0]
    geom = blockamr.Geometry(box, rb, 0, is_per)
    ba = blockamr.BoxArray(box)  # one box, no max_size split
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom, eb_factory=eb_factory), geom, ba, dm


def _inflow_bc(U_inf=1.0):
    return VectorBC(
        xlo=fixedValue([U_inf, 0.0, 0.0]),
        xhi=NeumannBC(),
        ylo=NeumannBC(),
        yhi=NeumannBC(),
    )


def _build_cellfield_with_inflow(init_value=0.5, U_inf=1.0):
    mesh, _, _, _ = _single_box_mesh()
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_inflow_bc(U_inf=U_inf)),
    )
    U.mf[0].set_val(init_value)
    U.fill_patch(0, 0.0)
    return U


def _init_uniform_U(solver, U_inf=1.0):
    new_arrs = []
    for arr in solver.U.mf[0].grown_arrays():
        a = np.asarray(arr).copy()
        a[..., 0] = U_inf
        a[..., 1] = 0.0
        a[..., 2] = 0.0
        new_arrs.append(jnp.asarray(a))
    solver.U.mf[0].copy_grown_arrays(new_arrs)
    solver.U.fill_patch(0, 0.0)


def _build_non_eb_solver_with_inflow(nx=16, ny=8, nz=4, U_inf=1.0,
                                     dt=0.001, nu=0.01):
    mesh, _, _, _ = _single_box_mesh(nx=nx, ny=ny, nz=nz)
    bc = _inflow_bc(U_inf=U_inf)
    solver = DSLIncompressibleSolver(
        mesh, nu=nu, dt=dt,
        fill_patch=FillPatchWithBC(bc),
    )
    _init_uniform_U(solver, U_inf=U_inf)
    return solver


def _build_eb_solver_with_inflow(nx=16, ny=8, nz=4, U_inf=1.0,
                                 cyl_radius=0.05, dt=0.001, nu=0.01,
                                 Lx=2.0, Ly=1.0, cyl_x_frac=0.2):
    """Single-box mesh with a small cylinder and inflow Dirichlet BCs.

    ``cyl_x_frac`` is the cylinder's x position as a fraction of Lx
    (default 0.2 → cylinder 4D from inflow when D=0.1, Lx=2). Increase
    it to put the cylinder farther downstream and reduce upstream
    pressure feedback through the elliptic projection.
    """
    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [Lx, Ly, Ly * nz / ny])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    dm = blockamr.DistributionMapping(ba)
    cyl = blockamr.EB2_CylinderIF(
        cyl_radius, 2, [cyl_x_frac * Lx, 0.5 * Ly, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    ebf = blockamr.make_eb_factory(geom, ba, dm)
    mesh = Mesh(ba, dm, geom, eb_factory=ebf)
    bc = _inflow_bc(U_inf=U_inf)
    solver = DSLIncompressibleSolver(
        mesh, nu=nu, dt=dt,
        fill_patch=FillPatchWithBC(bc),
    )
    _init_uniform_U(solver, U_inf=U_inf)
    return solver


def _build_eb_cellfield_with_inflow(nx=16, ny=16, nz=4, init_value=1.0):
    """Single-box EB cylinder + inflow CellField (no solver)."""
    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 0.25])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    dm = blockamr.DistributionMapping(ba)
    cyl = blockamr.EB2_CylinderIF(0.15, 2, [0.5, 0.5, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)
    ebf = blockamr.make_eb_factory(geom, ba, dm)
    mesh = Mesh(ba, dm, geom, eb_factory=ebf)
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_inflow_bc(U_inf=1.0)),
    )
    U.mf[0].set_val(init_value)
    U.fill_patch(0, 0.0)
    return mesh, U


# ===========================================================================
# B1 — Dirichlet inflow ghost fill on a non-EB mesh
# ===========================================================================


def test_dirichlet_xlo_ux_ghost_is_linear_extrapolation():
    """U_x ghost at xlo: 2 * 1.0 − 0.5 = 1.5."""
    U = _build_cellfield_with_inflow(init_value=0.5)
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(
        ghost_ux, 1.5, atol=1e-12,
        err_msg="xlo U_x Dirichlet ghost not 2*wall - interior",
    )


def test_dirichlet_xlo_uy_ghost_is_negative_half():
    """U_y ghost at xlo: 2 * 0.0 − 0.5 = -0.5 (BC value for U_y is 0)."""
    U = _build_cellfield_with_inflow(init_value=0.5)
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_uy = arr[0, ng:-ng, ng:-ng, 1]
    np.testing.assert_allclose(
        ghost_uy, -0.5, atol=1e-12,
        err_msg="xlo U_y Dirichlet ghost: expected 2*0 - 0.5 = -0.5",
    )


def test_dirichlet_consistent_state_when_init_matches_bc():
    """When the interior already equals the BC value, the ghost also
    equals the BC value (no jump). 2*1.0 - 1.0 = 1.0."""
    mesh, _, _, _ = _single_box_mesh()
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_inflow_bc(U_inf=1.0)),
    )
    new_arrs = []
    for arr in U.mf[0].grown_arrays():
        a = np.asarray(arr).copy()
        a[..., 0] = 1.0
        a[..., 1] = 0.0
        a[..., 2] = 0.0
        new_arrs.append(jnp.asarray(a))
    U.mf[0].copy_grown_arrays(new_arrs)
    U.fill_patch(0, 0.0)

    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(
        ghost_ux, 1.0, atol=1e-12,
        err_msg="consistent inflow state perturbed by BC fill",
    )


# ===========================================================================
# B2 — Neumann outflow ghost fill on a non-EB mesh
# ===========================================================================


def test_neumann_xhi_uniform_field_is_unchanged():
    """For a uniform field, Neumann ghost = interior = uniform value."""
    mesh, _, _, _ = _single_box_mesh()
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_inflow_bc()),
    )
    U.mf[0].set_val(0.7)
    U.fill_patch(0, 0.0)
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_ux_xhi = arr[-1, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(
        ghost_ux_xhi, 0.7, atol=1e-12,
        err_msg="xhi Neumann ghost not equal to interior",
    )


# ===========================================================================
# B2-slip — slipWall: zero normal velocity, Neumann tangential
# ===========================================================================
#
# A pure NeumannBC() on a vector field at a y wall does NOT enforce
# U_y = 0 — it sets ∂U_y/∂y = 0, which is "zero gradient" on the
# normal component, allowing fluid to leak through the wall. The
# proper free-slip / symmetry wall is mixed: Dirichlet 0 on the
# normal component (no penetration) + Neumann on the two tangential
# components (no shear).
#
# These tests verify the slipWall BC actually enforces both halves of
# this contract on a non-EB mesh, before we trust it in the cylinder
# example.


def _slip_wall_bc_in_y(U_inf=1.0):
    return VectorBC(
        xlo=fixedValue([U_inf, 0.0, 0.0]),
        xhi=NeumannBC(),
        ylo=slipWall(),
        yhi=slipWall(),
    )


def _build_cellfield_with_slip_walls(init_value=(0.7, 0.5, 0.0)):
    mesh, _, _, _ = _single_box_mesh()
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_slip_wall_bc_in_y(U_inf=1.0)),
    )
    new_arrs = []
    for arr in U.mf[0].grown_arrays():
        a = np.asarray(arr).copy()
        a[..., 0] = init_value[0]
        a[..., 1] = init_value[1]
        a[..., 2] = init_value[2]
        new_arrs.append(jnp.asarray(a))
    U.mf[0].copy_grown_arrays(new_arrs)
    U.fill_patch(0, 0.0)
    return U


def test_slipwall_ylo_uy_ghost_is_dirichlet_zero():
    """At a slip wall in y, U_y is the *normal* component → Dirichlet 0
    → ghost = 2*0 − 0.5 = -0.5 (consistent with no penetration through
    the wall)."""
    U = _build_cellfield_with_slip_walls(init_value=(0.7, 0.5, 0.0))
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_uy = arr[ng:-ng, 0, ng:-ng, 1]
    np.testing.assert_allclose(
        ghost_uy, -0.5, atol=1e-12,
        err_msg="ylo slip wall: U_y ghost not Dirichlet 0 (allows mass leak)",
    )


def test_slipwall_ylo_ux_ghost_is_neumann():
    """At a slip wall in y, U_x is *tangential* → Neumann (∂U_x/∂y=0)
    → ghost = interior. With interior=0.7, ghost should also be 0.7."""
    U = _build_cellfield_with_slip_walls(init_value=(0.7, 0.5, 0.0))
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_ux = arr[ng:-ng, 0, ng:-ng, 0]
    np.testing.assert_allclose(
        ghost_ux, 0.7, atol=1e-12,
        err_msg="ylo slip wall: U_x ghost not Neumann (would create shear)",
    )


def test_slipwall_yhi_uy_ghost_is_dirichlet_zero():
    """Same predicate at the high y face."""
    U = _build_cellfield_with_slip_walls(init_value=(0.7, 0.5, 0.0))
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_uy = arr[ng:-ng, -1, ng:-ng, 1]
    np.testing.assert_allclose(ghost_uy, -0.5, atol=1e-12)


def test_neumann_yhi_uniform_field_is_unchanged():
    """Neumann at yhi: ghost row = last interior row."""
    mesh, _, _, _ = _single_box_mesh()
    U = CellField(
        mesh, ncomp=3, ngrow=1, name="U",
        fill_patch=FillPatchWithBC(_inflow_bc()),
    )
    U.mf[0].set_val(0.3)
    U.fill_patch(0, 0.0)
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_yhi = arr[ng:-ng, -1, ng:-ng, 0]
    np.testing.assert_allclose(ghost_yhi, 0.3, atol=1e-12)


# ===========================================================================
# B3 — BC fill on an EB mesh: BC + eb_set_covered both run in the right order
# ===========================================================================


def test_eb_xlo_dirichlet_ghost_still_set_on_eb_mesh():
    """Inflow Dirichlet ghost at xlo is filled correctly even with EB."""
    _, U = _build_eb_cellfield_with_inflow(init_value=1.0)
    ng = U.mf[0].n_grow()
    arr = np.asarray(U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(
        ghost_ux, 1.0, atol=1e-12,
        err_msg="EB mesh: inflow Dirichlet ghost not preserved",
    )


def test_eb_covered_cells_zero_after_bc_fill():
    """Covered cells inside the cylinder are 0 after the BC fill."""
    mesh, U = _build_eb_cellfield_with_inflow(init_value=1.0)
    vf_np = np.asarray(mesh.vol_frac(0)[0])
    ng = U.mf[0].n_grow()
    u_valid = np.asarray(U.mf[0].arrays()[0])[ng:-ng, ng:-ng, ng:-ng, :]
    covered_mask = (vf_np == 0.0)
    n_covered = int(covered_mask.sum())
    assert n_covered > 0, "test setup needs some covered cells"
    for c in range(3):
        covered_vals = u_valid[..., c][covered_mask]
        np.testing.assert_allclose(
            covered_vals, 0.0, atol=1e-12,
            err_msg=f"covered cells of U[..., {c}] not zero after fill_patch",
        )


def test_eb_fluid_cells_unchanged_after_bc_fill():
    """Fluid cells (vol_frac == 1) keep the initial value 1.0."""
    mesh, U = _build_eb_cellfield_with_inflow(init_value=1.0)
    ng = U.mf[0].n_grow()
    u_valid = np.asarray(U.mf[0].arrays()[0])[ng:-ng, ng:-ng, ng:-ng, :]
    vf_np = np.asarray(mesh.vol_frac(0)[0])
    fluid_mask = (vf_np == 1.0)
    for c in range(3):
        np.testing.assert_allclose(
            u_valid[..., c][fluid_mask], 1.0, atol=1e-12,
            err_msg=f"fluid cells of U[..., {c}] perturbed by fill_patch",
        )


# ===========================================================================
# B4 — One solver step preserves the inflow Dirichlet ghost (non-EB)
# ===========================================================================
#
# This is the smoking-gun test for the current Re=20 blow-up. If
# ``solver.step()`` clobbers the xlo Dirichlet ghost, the inflow
# erodes step by step and the simulation drifts unboundedly — the
# exact symptom we observed.
#
# The non-EB variant runs first to isolate the BC bug from the EB
# code path. The B4-EB variant follows.


def test_step_xlo_ghost_after_fill_only_non_eb():
    """Sanity: before any step, the xlo ghost is the consistent value 1.0
    (interior=1, dirichlet=1, ghost = 2*1−1 = 1)."""
    solver = _build_non_eb_solver_with_inflow()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(ghost_ux, 1.0, atol=1e-12)


def test_step_xlo_ghost_preserved_after_one_step_non_eb():
    """**B4 hot test (non-EB).** After one ``solver.step()`` from a
    uniform IC with inflow Dirichlet, the xlo ghost still equals 1.0."""
    solver = _build_non_eb_solver_with_inflow()
    solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(ghost_ux - 1.0)))
    assert max_dev < 1e-6, (
        f"xlo Dirichlet ghost drifted from 1.0 after one step: "
        f"max|ghost_ux - 1| = {max_dev:.3e}"
    )


def test_step_xlo_first_interior_close_to_uinf_after_one_step_non_eb():
    """The first interior cell column at xlo should not deviate much
    from U_inf after one step. Tolerance is loose because the predictor
    introduces O(dt) numerical diffusion, but a value far from 1
    (|Δ| > 0.1) means the corrector or projection is overwriting it."""
    solver = _build_non_eb_solver_with_inflow()
    solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    first_int_ux = arr[ng, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(first_int_ux - 1.0)))
    assert max_dev < 0.1, (
        f"xlo first interior U_x deviates from 1 by {max_dev:.3e} "
        "after one step (should be O(dt) ~ 1e-3)"
    )


# ===========================================================================
# B4-EB — One step on an EB mesh preserves the inflow Dirichlet ghost
# ===========================================================================
#
# These tests are the *EB-specific* counterpart of B4. The non-EB B4
# already passes (predictor + nodal corrector preserve the BC). The
# EB code path takes a *different* corrector (``_pressure_correct_eb``)
# so this is where the bug — if there is one in the corrector — would
# show up.


def test_step_xlo_ghost_consistent_at_t0_eb():
    """Sanity: the EB solver starts from a consistent state where the
    xlo ghost equals the BC value (interior=1, dirichlet=1)."""
    solver = _build_eb_solver_with_inflow()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    np.testing.assert_allclose(ghost_ux, 1.0, atol=1e-12)


def test_step_xlo_ghost_preserved_after_one_step_eb():
    """**Smoking-gun test for the Re=20 blow-up.** After one
    ``solver.step()`` on an EB mesh from a uniform IC with inflow
    Dirichlet, the xlo ghost should still equal 1.0.

    If this *fails* while the non-EB equivalent passes, the bug is
    localised to the EB code path (``_pressure_correct_eb`` or the
    EB momentum predictor through volfrac masking).
    """
    solver = _build_eb_solver_with_inflow()
    solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(ghost_ux - 1.0)))
    assert max_dev < 1e-6, (
        f"EB mesh: xlo Dirichlet ghost drifted from 1.0 after one step: "
        f"max|ghost_ux - 1| = {max_dev:.3e}. The non-EB equivalent passes, "
        "so the bug is in the EB code path."
    )


def test_step_xlo_first_interior_close_to_uinf_after_one_step_eb():
    """First interior column at xlo should be close to U_inf after one
    EB step. Tolerance loose (predictor diffusion + EB cut-cell first-
    order error)."""
    solver = _build_eb_solver_with_inflow()
    solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    first_int_ux = arr[ng, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(first_int_ux - 1.0)))
    assert max_dev < 0.1, (
        f"EB mesh: first interior U_x at xlo deviates from 1 by "
        f"{max_dev:.3e} after one step"
    )


# ===========================================================================
# B4-N — Multi-step BC erosion + max|U| stability on EB
# ===========================================================================
#
# A single step might not show the bug (it's slow drift). These tests
# run N steps and check whether the inflow BC erodes or max|U| diverges.
# Both directly probe the symptom of the Re=20 blow-up at small N for
# fast feedback.


def test_step_xlo_ghost_preserved_after_10_steps_eb_close_inflow():
    """After 10 EB steps with the cylinder 4D from inflow, the inflow
    ghost is perturbed by upstream pressure feedback through the
    elliptic projection. The threshold (5%) is loose because the
    inflow is intentionally close — see the *_far_inflow* variant
    below for the tight version.
    """
    solver = _build_eb_solver_with_inflow(dt=0.001, cyl_x_frac=0.2)
    for _ in range(10):
        solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(ghost_ux - 1.0)))
    assert max_dev < 5e-2, (
        f"EB inflow ghost eroded after 10 steps (close inflow): max|Δ| = {max_dev:.3e}"
    )


def test_step_xlo_ghost_preserved_after_10_steps_eb_far_inflow():
    """Same as the close-inflow variant but with the cylinder placed
    much farther downstream (cyl_x = 0.8*Lx, ~16D from inflow) so that
    the upstream pressure feedback is negligible. This isolates the
    *physical* feedback effect from any remaining numerical drift.

    With buggy face-averaging this would still fail at ~2.8e-2.
    With the EB MLNodeLaplacian fix this should pass at < 1e-3.
    """
    solver = _build_eb_solver_with_inflow(
        dt=0.001, cyl_radius=0.05, cyl_x_frac=0.8)
    for _ in range(10):
        solver.step()
    ng = solver.U.mf[0].n_grow()
    arr = np.asarray(solver.U.mf[0].arrays()[0])
    ghost_ux = arr[0, ng:-ng, ng:-ng, 0]
    max_dev = float(np.max(np.abs(ghost_ux - 1.0)))
    assert max_dev < 1e-3, (
        f"EB inflow ghost eroded after 10 steps (far inflow): max|Δ| = {max_dev:.3e}. "
        "Far-inflow drift indicates a real bug, not pressure feedback."
    )


def test_step_max_velocity_bounded_after_10_steps_eb():
    """After 10 EB steps from a uniform IC, max|U| should not exceed
    the constriction bound 1/(1-D/Ly) ≈ 1.11 by more than a small
    margin. With cyl_radius=0.05 and Ly=1, the analytic max from
    continuity is 1.0 / (1 - 2*0.05) = 1.111."""
    solver = _build_eb_solver_with_inflow(cyl_radius=0.05, dt=0.001)
    max_vels = []
    for _ in range(10):
        solver.step()
        max_vels.append(float(solver._max_velocity()))
    final_max = max_vels[-1]
    assert final_max < 1.5, (
        f"max|U| reached {final_max:.4f} after 10 steps (should be ≤ ~1.2). "
        f"Trajectory: {[f'{v:.3f}' for v in max_vels]}"
    )


def test_eb_inflow_outflow_mass_conservation_after_50_steps():
    """First-vs-last column mass flux check (necessary, not sufficient)."""
    solver = _build_eb_solver_with_inflow(
        nx=32, ny=16, nz=4, cyl_radius=0.05, dt=0.001)
    for _ in range(50):
        solver.step()

    mf = solver.U.mf[0]
    ng = mf.n_grow()
    arr = np.asarray(mf.arrays()[0])
    nx = arr.shape[0] - 2 * ng

    ux_inflow  = arr[ng,            ng:-ng, ng:-ng, 0]
    ux_outflow = arr[ng + nx - 1,   ng:-ng, ng:-ng, 0]

    mass_in  = float(ux_inflow.sum())
    mass_out = float(ux_outflow.sum())
    ratio = mass_out / mass_in if mass_in != 0 else float("inf")

    assert 0.9 < ratio < 1.1, (
        f"mass conservation failed: in={mass_in:.4f}, out={mass_out:.4f}, "
        f"ratio={ratio:.4f} (expect ~1.0). Slip walls or projection "
        "are letting mass through the lateral boundaries."
    )


def test_eb_streamwise_mass_flux_constant_along_x_high_res():
    """Resolution-matched to the user's failing example
    (--nz 8 --ncell 128 → nx=256, ny=128, nz=8)."""
    nx, ny, nz = 256, 128, 8
    U_inf = 1.0
    Ly = 1.0
    solver = _build_eb_solver_with_inflow(
        nx=nx, ny=ny, nz=nz, U_inf=U_inf, cyl_radius=0.05, dt=0.001)
    for _ in range(50):
        solver.step()

    mf = solver.U.mf[0]
    ng = mf.n_grow()
    arr = np.asarray(mf.arrays()[0])
    nx_v = arr.shape[0] - 2 * ng
    ny_v = arr.shape[1] - 2 * ng
    nz_v = arr.shape[2] - 2 * ng
    dy = Ly / ny_v

    u_valid = arr[ng:ng + nx_v, ng:ng + ny_v, ng:ng + nz_v, 0]
    flux_per_col = u_valid.sum(axis=(1, 2)) * dy / nz_v
    expected = U_inf * Ly
    rel_err = np.abs(flux_per_col - expected) / expected
    max_rel_err = float(rel_err.max())

    assert max_rel_err < 0.05, (
        f"high-res mass flux not constant in x: max rel err = "
        f"{max_rel_err:.4f}. Profile (in / mid / out): "
        f"{flux_per_col[0]:.4f} / {flux_per_col[nx_v // 2]:.4f} / "
        f"{flux_per_col[-1]:.4f}. Expected {expected:.4f} everywhere."
    )


def test_eb_streamwise_mass_flux_constant_along_x_after_50_steps():
    """**Strict mass-conservation test.** The streamwise mass flux
    ``∫U_x(x, y) dy`` must equal the inflow flux ``U_inf · Ly`` at
    *every* x station — not just the first and last.

    If mass leaks through the y walls or the projection introduces
    spurious sources, the per-x flux drifts as you move downstream
    (e.g. monotone decrease toward zero, the symptom in the user's
    rendered cylinder image). This test integrates U_x over y at each
    x column and checks the entire profile is within 5% of U_inf · Ly.
    """
    nx, ny = 32, 16
    U_inf = 1.0
    Ly = 1.0
    solver = _build_eb_solver_with_inflow(
        nx=nx, ny=ny, nz=4, U_inf=U_inf, cyl_radius=0.05, dt=0.001)
    for _ in range(50):
        solver.step()

    mf = solver.U.mf[0]
    ng = mf.n_grow()
    arr = np.asarray(mf.arrays()[0])
    nx_v = arr.shape[0] - 2 * ng
    ny_v = arr.shape[1] - 2 * ng
    nz_v = arr.shape[2] - 2 * ng
    dy = Ly / ny_v

    # Per-x-column mass flux ∫U_x dy averaged over z (z is periodic so
    # any z slice is equivalent — we mean over z to denoise)
    u_valid = arr[ng:ng + nx_v, ng:ng + ny_v, ng:ng + nz_v, 0]
    flux_per_col = u_valid.sum(axis=(1, 2)) * dy / nz_v   # shape (nx_v,)

    expected = U_inf * Ly
    rel_err = np.abs(flux_per_col - expected) / expected
    max_rel_err = float(rel_err.max())
    arg_worst = int(rel_err.argmax())

    assert max_rel_err < 0.05, (
        f"streamwise mass flux is not constant in x: "
        f"max relative error = {max_rel_err:.4f} at column {arg_worst} of {nx_v}, "
        f"flux there = {flux_per_col[arg_worst]:.4f}, expected = {expected:.4f}. "
        f"Profile (first / mid / last): {flux_per_col[0]:.4f} / "
        f"{flux_per_col[nx_v // 2]:.4f} / {flux_per_col[-1]:.4f}"
    )


def test_step_max_velocity_bounded_after_50_steps_eb():
    """A longer-horizon variant. 50 steps at dt=0.001 ⇒ t=0.05,
    which is ~5 cylinder passes. max|U| should be approaching steady
    ~1.1 not diverging."""
    solver = _build_eb_solver_with_inflow(cyl_radius=0.05, dt=0.001)
    max_vels = []
    for _ in range(50):
        solver.step()
        max_vels.append(float(solver._max_velocity()))
    final_max = max_vels[-1]
    assert final_max < 1.5, (
        f"max|U| diverged after 50 steps: {final_max:.4f}. "
        f"Final 5 values: {[f'{v:.3f}' for v in max_vels[-5:]]}"
    )
