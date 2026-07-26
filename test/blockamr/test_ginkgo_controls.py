# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Solver controls & diagnostics of the matrix-free Ginkgo solvers.

Covers the absolute-tolerance stop (``atol``), the ``converged`` flag and the
per-iteration residual history (``res_history``) returned by the persistent
face-coefficient solvers and the one-shot ``ginkgo_solve``. The model problem
is the periodic Helmholtz operator (phi - laplacian phi): diagonal source
alpha=1 with symmetric face coefficients -1/dx^2.
"""

import math

import numpy as np
import pytest

import blockamr


def _make_periodic_mesh(n):
    """Single-box periodic mesh on [0,1]^3 with n cells per side."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


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


def _random_rhs(ba, dm, seed=42):
    """Cell MultiFab with seeded random values — full spectrum, so CG must iterate."""
    rng = np.random.default_rng(seed)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        rhs.copy_from(mfi, arr)
    return rhs


def _max_abs_diff(a, b):
    """Max-norm difference between the valid regions of two cell MultiFabs."""
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


def _helmholtz_coeffs(geom, ba, dm, n):
    """alpha=1 cell source + symmetric -1/dx^2 face coeffs (periodic Helmholtz)."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 1.0)
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
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=executor, **kwargs
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


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_atol_stops_earlier(blockamr_session, executor):
    """With a very tight rtol, a moderate atol stops the solve strictly earlier.

    The reported final true residual ||b - A x|| then satisfies the absolute
    criterion the solver stopped on.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    atol = 1e-3

    s_base = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-14
    )
    stats_base = s_base.solve(rhs, _zero_sol(ba, dm))

    s_atol = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-14, atol=atol
    )
    stats_atol = s_atol.solve(rhs, _zero_sol(ba, dm))

    assert stats_atol["num_iters"] < stats_base["num_iters"], (
        f"atol solve took {stats_atol['num_iters']} iters, "
        f"rtol-only took {stats_base['num_iters']}"
    )
    assert stats_atol["converged"] is True
    assert stats_atol["res_norm"] <= atol, (
        f"true residual {stats_atol['res_norm']} exceeds atol {atol}"
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("cls", ["FaceCoeffSolver", "FaceCoeffCsrSolver"])
def test_res_history_matches_num_iters(blockamr_session, cls, executor):
    """len(res_history) == num_iters + 1 (initial residual + one entry per iteration).

    The history starts at ||rhs|| (zero initial guess), decreases from first to
    last entry, and resets between calls on a persistent solver.
    """
    N = 16
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s = _make_solver_or_skip(cls, coeffs, geom, executor, solver="cg", max_iter=500, rtol=1e-10)
    stats = s.solve(rhs, _zero_sol(ba, dm))

    hist = stats["res_history"]
    assert len(hist) == stats["num_iters"] + 1
    assert all(math.isfinite(v) for v in hist)
    assert hist[-1] < hist[0], f"history not decreasing: first {hist[0]}, last {hist[-1]}"
    # Zero initial guess -> the first logged residual norm is ||rhs||.
    rhs_norm = math.sqrt(sum(float(np.sum(b**2)) for b in
                             (rhs.copy_to_host(mfi) for mfi in blockamr.MFIterator(rhs))))
    assert hist[0] == pytest.approx(rhs_norm, rel=1e-10)

    # Per-call history: a second solve reports its own iterations, not a growing log.
    stats2 = s.solve(rhs, _zero_sol(ba, dm))
    assert len(stats2["res_history"]) == stats2["num_iters"] + 1


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_converged_flag(blockamr_session, executor):
    """converged is True on a normal solve, False when max_iter cuts the solve short."""
    N = 16
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s_ok = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, solver="cg", max_iter=500, rtol=1e-10
    )
    stats_ok = s_ok.solve(rhs, _zero_sol(ba, dm))
    assert stats_ok["converged"] is True
    assert stats_ok["num_iters"] < 500

    s_cut = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, solver="cg", max_iter=2, rtol=1e-12
    )
    stats_cut = s_cut.solve(rhs, _zero_sol(ba, dm))
    assert stats_cut["converged"] is False
    assert stats_cut["num_iters"] == 2


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_default_no_atol_unchanged(blockamr_session, executor):
    """Without atol the solve behaves as before: mf and csr solvers agree.

    Same matrix, same stopping rule (Iteration + rhs_norm ResidualNorm), so the
    two solutions match and both converge.
    """
    N = 16
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s_mf = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, solver="cg", max_iter=500, rtol=1e-10
    )
    sol_mf = _zero_sol(ba, dm)
    stats_mf = s_mf.solve(rhs, sol_mf)

    s_csr = _make_solver_or_skip(
        "FaceCoeffCsrSolver", coeffs, geom, executor, solver="cg", max_iter=500, rtol=1e-10
    )
    sol_csr = _zero_sol(ba, dm)
    stats_csr = s_csr.solve(rhs, sol_csr)

    assert stats_mf["converged"] is True
    assert stats_csr["converged"] is True
    assert stats_mf["res_norm"] < 1e-6
    assert stats_csr["res_norm"] < 1e-6
    max_diff = _max_abs_diff(sol_mf, sol_csr)
    assert max_diff < 1e-6, f"Max |sol_mf - sol_csr| = {max_diff} exceeds 1e-6"


def test_ginkgo_solve_controls(blockamr_session):
    """One-shot ginkgo_solve: converged flag, res_history, and the atol stop.

    MLPoisson Dirichlet problem as in test_ginkgo_poisson; the reported
    res_norm is the true residual of the correction system, which equals the
    original-system residual, so it must sit below atol when atol stops the solve.
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 16
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    lp.set_level_bc(0, None)

    rhs = _random_rhs(ba, dm)

    sol = _zero_sol(ba, dm)
    try:
        stats = blockamr.ginkgo_solve(lp, sol, rhs, max_iter=2000, rtol=1e-10)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise
    assert stats["converged"] is True
    hist = stats["res_history"]
    assert len(hist) == stats["num_iters"] + 1
    assert hist[-1] < hist[0]

    # atol stops earlier than the rtol=1e-14 solve and meets the absolute bound.
    atol = 1e-3
    stats_tight = blockamr.ginkgo_solve(lp, _zero_sol(ba, dm), rhs, max_iter=2000, rtol=1e-14)
    stats_atol = blockamr.ginkgo_solve(
        lp, _zero_sol(ba, dm), rhs, max_iter=2000, rtol=1e-14, atol=atol
    )
    assert stats_atol["num_iters"] < stats_tight["num_iters"]
    assert stats_atol["converged"] is True
    assert stats_atol["res_norm"] <= atol

    # max_iter too small -> not converged.
    stats_cut = blockamr.ginkgo_solve(lp, _zero_sol(ba, dm), rhs, max_iter=2, rtol=1e-12)
    assert stats_cut["converged"] is False
    assert stats_cut["num_iters"] == 2
