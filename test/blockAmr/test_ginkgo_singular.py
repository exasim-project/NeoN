# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Singular-system (constant-nullspace) handling of the persistent Ginkgo solvers.

The fully-periodic pure Poisson pressure equation (alpha=0, all face coeffs
-1/dx^2) is singular: constants are in the nullspace, so the system is only
solvable for a mean-zero rhs and the solution is defined up to a constant.
MLMG handles this internally; the matrix-free path exposes it as the
``project_nullspace`` constructor kwarg, which projects the rhs and initial
guess mean-zero before the Krylov solve and returns the mean-zero solution.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor


def _make_periodic_mesh(n):
    """Single-box periodic mesh on [0,1]^3 with n cells per side."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _meanzero_values(n, seed=42):
    """Seeded random cell values with the mean subtracted exactly.

    Random data spans the whole spectrum, so CG must genuinely iterate — a
    smooth single-mode rhs is a Laplacian eigenvector and converges in one step.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, n, n))
    return v - v.mean()


def _fill_values(mf, values, offset=0.0):
    """Fill the (single-box) cell MultiFab with values + offset."""
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values + offset
        mf.copy_from(mfi, arr)


def _const_cell(ba, dm, value):
    """Cell-centred MultiFab (no ghost) filled with a constant."""
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(geom, dm, d, n, value):
    """Face-centred MultiFab in direction d filled with a constant."""
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(n)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _to_host(mf):
    """Flattened valid-region values of a cell MultiFab."""
    return np.concatenate(
        [mf.copy_to_host(mfi)[:, :, :, 0].ravel() for mfi in blockamr.MFIterator(mf)]
    )


def _poisson_coeffs(geom, ba, dm, n):
    """Singular pure Poisson: alpha=0 cell source + symmetric -1/dx^2 face coeffs."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 0.0)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _make_solver_or_skip(cls, coeffs, geom, executor, **kwargs):
    """Construct a persistent solver, skipping if Ginkgo/CUDA are unavailable."""
    if not hasattr(blockamr, cls):
        pytest.skip(f"blockamr.{cls} binding not available")
    alpha, fx, fy, fz = coeffs
    try:
        return getattr(blockamr, cls)(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=gko_executor(executor), **kwargs
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _mlmg_poisson_reference(geom, ba, dm, n, rhs):
    """Solve -laplacian(phi) = rhs with MLABecLaplacian(a=0, b=1) + MLMG.

    MLMG handles the singular fully-periodic problem internally; the returned
    solution's constant offset is arbitrary, so callers compare mean-subtracted.
    """
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3,
        [blockamr.LinOpBCType.Periodic] * 3,
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(0.0, 1.0)  # a=0 -> pure -div(grad phi): singular
    abec.set_a_coeffs(0, _const_cell(ba, dm, 0.0))
    abec.set_b_coeffs(
        0,
        _const_face(geom, dm, 0, n, 1.0),
        _const_face(geom, dm, 1, n, 1.0),
        _const_face(geom, dm, 2, n, 1.0),
    )
    sol_ref = _zero_sol(ba, dm)
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-11, 1e-13)
    return _to_host(sol_ref)


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_singular_poisson_projected_matches_mlmg(blockamr_session, executor):
    """Cases (a)+(b): mean-zero rhs, project_nullspace=True.

    The solve converges, the returned solution is mean-zero, and after
    mean-subtracting both it matches the MLMG solution of the same singular
    operator.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _poisson_coeffs(geom, ba, dm, N)

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_values(rhs, _meanzero_values(N))

    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=2000,
        rtol=1e-11,
        project_nullspace=True,
    )
    sol = _zero_sol(ba, dm)
    stats = s.solve(rhs, sol)

    assert stats["converged"] is True
    assert stats["num_iters"] > 1, "multi-mode rhs should take more than one CG iteration"
    sol_h = _to_host(sol)
    assert abs(float(np.mean(sol_h))) < 1e-10, f"solution mean {np.mean(sol_h)} not ~0"

    ref_h = _mlmg_poisson_reference(geom, ba, dm, N, rhs)
    diff = (sol_h - sol_h.mean()) - (ref_h - ref_h.mean())
    max_diff = float(np.max(np.abs(diff)))
    assert max_diff < 1e-6, f"Max |sol - sol_mlmg| = {max_diff} exceeds 1e-6"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_singular_poisson_nonzero_mean_rhs(blockamr_session, executor):
    """Case (c): rhs with a nonzero mean (inconsistent system).

    project_nullspace=True removes the mean, so the solve still converges and
    (mean-subtracted) matches the MLMG solution of the mean-projected rhs.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _poisson_coeffs(geom, ba, dm, N)
    values = _meanzero_values(N)

    rhs_off = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_values(rhs_off, values, offset=0.7)

    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=2000,
        rtol=1e-11,
        project_nullspace=True,
    )
    sol = _zero_sol(ba, dm)
    stats = s.solve(rhs_off, sol)

    assert stats["converged"] is True
    sol_h = _to_host(sol)
    assert abs(float(np.mean(sol_h))) < 1e-10

    # MLMG reference on the explicitly mean-projected rhs.
    rhs_proj = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_values(rhs_proj, values)  # the +0.7 offset removed exactly
    ref_h = _mlmg_poisson_reference(geom, ba, dm, N, rhs_proj)
    diff = (sol_h - sol_h.mean()) - (ref_h - ref_h.mean())
    max_diff = float(np.max(np.abs(diff)))
    assert max_diff < 1e-6, f"Max |sol - sol_mlmg| = {max_diff} exceeds 1e-6"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_singular_poisson_without_projection_fails(blockamr_session, executor):
    """Case (d) sanity: same inconsistent problem, project_nullspace=False.

    The rhs has a component in the nullspace of A^T, so the residual can never
    reach rtol*||rhs||: the flag demonstrably changes the outcome. Asserted
    loosely (no convergence to tolerance), not pinning the exact behaviour of a
    singular solve.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _poisson_coeffs(geom, ba, dm, N)

    rhs_off = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_values(rhs_off, _meanzero_values(N), offset=0.7)

    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=500,
        rtol=1e-11,
        project_nullspace=False,
    )
    stats = s.solve(rhs_off, _zero_sol(ba, dm))

    # The constant rhs component (0.7 per cell) is unreachable: ||r|| >= 0.7*sqrt(n).
    assert stats["converged"] is False or stats["res_norm"] > 1e-3, (
        f"unprojected singular solve unexpectedly converged: {stats}"
    )
