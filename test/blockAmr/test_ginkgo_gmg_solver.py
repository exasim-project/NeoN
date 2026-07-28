# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""M1 native stationary GMG solver (``solver="gmg"``).

Runs the GMG V-cycle as a standalone Richardson solver ``x <- x + V(b - A x)``
until tolerance — no Ginkgo Krylov object, the whole loop on AMReX MultiFabs —
instead of using it as a CG preconditioner (``precond="gmg"``). Because a
standalone V-cycle needs an accurate coarsest-grid solve (unlike the CG
preconditioner, where CG mops up the low-frequency error), these tests raise
``gmg_coarsest_sweeps``. Model problem is the same periodic Helmholtz with a
seeded random rhs as ``test_ginkgo_gmg_knobs.py`` / ``test_ginkgo_gmg_fp32.py``.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor


def _make_mesh(n):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(geom, dm, d, n, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(n)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _random_rhs(ba, dm, seed=42):
    rng = np.random.default_rng(seed)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        rhs.copy_from(mfi, arr)
    return rhs


def _helmholtz_coeffs(geom, ba, dm, n):
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 1.0)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _make_solver_or_skip(coeffs, geom, executor, **kwargs):
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")
    alpha, fx, fy, fz = coeffs
    try:
        return blockamr.FaceCoeffSolver(
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
    sol.set_val(0.0)
    return sol


def _sol_to_host(mf):
    return np.concatenate(
        [mf.copy_to_host(mfi)[:, :, :, 0].ravel() for mfi in blockamr.MFIterator(mf)]
    )


# A standalone V-cycle needs an accurate bottom solve; the CG preconditioner does
# not (matches the bench_solvers.py `gmg` method).
def _coarsest(smoother):
    return 160 if smoother == "chebyshev" else 100


# ---------------------------------------------------------------------------
# Convergence + agreement with the CG path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("smoother", ["rbgs", "chebyshev"])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_gmg_solver_converges_and_matches_cg(blockamr_session, executor, smoother, precision):
    """solver="gmg" converges to rtol and agrees with the CG path to < 1e-6."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    co = _coarsest(smoother)

    # CG reference (precond="gmg", same V-cycle).
    s_cg = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=200, rtol=1e-10, precond="gmg",
        gmg_smoother=smoother, gmg_precision=precision, gmg_coarsest_sweeps=co,
    )
    sol_cg = _zero_sol(ba, dm)
    st_cg = s_cg.solve(rhs, sol_cg)
    assert st_cg["converged"] is True

    # Native stationary GMG solver.
    s_gmg = _make_solver_or_skip(
        coeffs, geom, executor, solver="gmg", max_iter=200, rtol=1e-10,
        gmg_smoother=smoother, gmg_precision=precision, gmg_coarsest_sweeps=co,
    )
    sol_gmg = _zero_sol(ba, dm)
    st_gmg = s_gmg.solve(rhs, sol_gmg)

    # Converges, with the final FP64 residual actually meeting the tolerance.
    assert st_gmg["converged"] is True
    assert st_gmg["res_norm"] < 1e-6
    # num_iters counts V-cycles; a real MG stays within a handful of CG iters.
    assert st_gmg["num_iters"] <= st_cg["num_iters"] + 8

    # Both solve the SAME system, so the converged solutions agree.
    diff = np.max(np.abs(_sol_to_host(sol_cg) - _sol_to_host(sol_gmg)))
    assert diff < 1e-6, f"gmg vs cg solution disagree: max|Δ|={diff:.2e}"


# ---------------------------------------------------------------------------
# Warm start (persistent-solver contract: incoming sol seeds the guess)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_solver_warm_start(blockamr_session, executor):
    """A second solve seeded with the converged solution needs <=1 cycle."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="gmg", max_iter=200, rtol=1e-10,
        gmg_coarsest_sweeps=100,
    )
    sol = _zero_sol(ba, dm)
    st_cold = s.solve(rhs, sol)
    assert st_cold["converged"] is True and st_cold["num_iters"] >= 2
    # sol now holds the converged solution; re-solving from it is essentially free.
    st_warm = s.solve(rhs, sol)
    assert st_warm["converged"] is True
    assert st_warm["num_iters"] <= 1, f"warm start did not help: {st_warm['num_iters']} cycles"


# ---------------------------------------------------------------------------
# max_iter bounds the cycle count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_solver_max_iter_respected(blockamr_session, executor):
    """max_iter caps the V-cycle count; short of convergence it reports not-converged."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="gmg", max_iter=3, rtol=1e-12,
        gmg_coarsest_sweeps=100,
    )
    st = s.solve(rhs, _zero_sol(ba, dm))
    assert st["num_iters"] == 3
    assert st["converged"] is False


# ---------------------------------------------------------------------------
# Stats dict parity with the CG path
# ---------------------------------------------------------------------------
def test_gmg_solver_stats_keys_match_cg(blockamr_session):
    """The stats dict has the same keys as the CG path."""
    N = 16
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s_cg = _make_solver_or_skip(
        coeffs, geom, "reference", solver="cg", max_iter=100, rtol=1e-10, precond="gmg",
    )
    s_gmg = _make_solver_or_skip(
        coeffs, geom, "reference", solver="gmg", max_iter=100, rtol=1e-10,
        gmg_coarsest_sweeps=60,
    )
    st_cg = s_cg.solve(rhs, _zero_sol(ba, dm))
    st_gmg = s_gmg.solve(rhs, _zero_sol(ba, dm))
    assert set(st_gmg.keys()) == set(st_cg.keys())


# ---------------------------------------------------------------------------
# Asymmetric sweeps are legitimate in gmg mode (not CG) -> no warning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_solver_asymmetric_sweeps_no_warning(blockamr_session, executor, capfd):
    """solver="gmg" is not CG, so unequal pre/post sweeps do NOT warn and still solve."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="gmg", max_iter=200, rtol=1e-10,
        gmg_pre_sweeps=2, gmg_post_sweeps=1, gmg_coarsest_sweeps=100,
    )
    out = capfd.readouterr()
    assert "non-symmetric" not in (out.err + out.out), "gmg mode must not warn on pre != post"
    stats = s.solve(rhs, _zero_sol(ba, dm))
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6
