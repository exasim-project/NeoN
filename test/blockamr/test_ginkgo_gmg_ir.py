# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Ginkgo iterative-refinement GMG solver (``solver="ir"``).

The Ginkgo-idiomatic twin of the native ``solver="gmg"`` loop: a
``gko::solver::Ir<double>`` (iterative refinement, relaxation 1.0) whose system
matrix is the matrix-free ``FaceCoeffOp`` and whose inner solver is the generated
GMG V-cycle LinOp. Mathematically it is the SAME Richardson iteration
``x <- x + V(b - A x)`` as ``solver="gmg"``, but driven through Ginkgo (Dense
pack/unpack + per-iteration LinOp crossings) instead of natively on AMReX
MultiFabs. Like ``solver="gmg"`` a standalone V-cycle needs an accurate
coarsest-grid solve, so these tests raise ``gmg_coarsest_sweeps`` identically.
Model problem is the same periodic Helmholtz with a seeded random rhs as
``test_ginkgo_gmg_solver.py``.
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


# A standalone V-cycle (native gmg OR Ginkgo Ir) needs an accurate bottom solve;
# the CG preconditioner does not (matches the bench_solvers.py `gmg`/`gmg-ir`).
def _coarsest(smoother):
    return 160 if smoother == "chebyshev" else 100


# ---------------------------------------------------------------------------
# Convergence + agreement with the native gmg loop and the CG path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("smoother", ["rbgs", "chebyshev"])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_ir_solver_converges_and_matches(blockamr_session, executor, smoother, precision):
    """solver="ir" converges to rtol and agrees with both gmg and CG to < 1e-6,
    with an iteration count within ~2 of the native gmg loop."""
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

    # Native stationary GMG solver (the loop Ir mirrors).
    s_gmg = _make_solver_or_skip(
        coeffs, geom, executor, solver="gmg", max_iter=200, rtol=1e-10,
        gmg_smoother=smoother, gmg_precision=precision, gmg_coarsest_sweeps=co,
    )
    sol_gmg = _zero_sol(ba, dm)
    st_gmg = s_gmg.solve(rhs, sol_gmg)
    assert st_gmg["converged"] is True

    # Ginkgo iterative-refinement GMG solver.
    s_ir = _make_solver_or_skip(
        coeffs, geom, executor, solver="ir", max_iter=200, rtol=1e-10,
        gmg_smoother=smoother, gmg_precision=precision, gmg_coarsest_sweeps=co,
    )
    sol_ir = _zero_sol(ba, dm)
    st_ir = s_ir.solve(rhs, sol_ir)

    # Converges, with the final residual actually meeting the tolerance.
    assert st_ir["converged"] is True
    assert st_ir["res_norm"] < 1e-6

    # Same Richardson iteration as the native gmg loop -> cycle count within ~2.
    assert abs(st_ir["num_iters"] - st_gmg["num_iters"]) <= 2, (
        f"ir {st_ir['num_iters']} vs gmg {st_gmg['num_iters']} cycles diverge"
    )

    # All three solve the SAME system, so the converged solutions agree.
    ir_host = _sol_to_host(sol_ir)
    d_gmg = np.max(np.abs(ir_host - _sol_to_host(sol_gmg)))
    d_cg = np.max(np.abs(ir_host - _sol_to_host(sol_cg)))
    assert d_gmg < 1e-6, f"ir vs gmg solution disagree: max|Δ|={d_gmg:.2e}"
    assert d_cg < 1e-6, f"ir vs cg solution disagree: max|Δ|={d_cg:.2e}"


# ---------------------------------------------------------------------------
# max_iter bounds the cycle count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_ir_solver_max_iter_respected(blockamr_session, executor):
    """max_iter caps the Ir iteration count; short of convergence -> not-converged."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="ir", max_iter=3, rtol=1e-12,
        gmg_coarsest_sweeps=100,
    )
    st = s.solve(rhs, _zero_sol(ba, dm))
    assert st["num_iters"] == 3
    assert st["converged"] is False


# ---------------------------------------------------------------------------
# Stats dict parity with the CG path
# ---------------------------------------------------------------------------
def test_ir_solver_stats_keys_match_cg(blockamr_session):
    """The stats dict has the same keys as the CG path."""
    N = 16
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s_cg = _make_solver_or_skip(
        coeffs, geom, "reference", solver="cg", max_iter=100, rtol=1e-10, precond="gmg",
    )
    s_ir = _make_solver_or_skip(
        coeffs, geom, "reference", solver="ir", max_iter=100, rtol=1e-10,
        gmg_coarsest_sweeps=60,
    )
    st_cg = s_cg.solve(rhs, _zero_sol(ba, dm))
    st_ir = s_ir.solve(rhs, _zero_sol(ba, dm))
    assert set(st_ir.keys()) == set(st_cg.keys())


# ---------------------------------------------------------------------------
# solver="ir" + precond_mlmg is rejected (mirrors solver="gmg")
# ---------------------------------------------------------------------------
def test_ir_solver_rejects_precond_mlmg(blockamr_session):
    """solver="ir" cannot be combined with precond_mlmg."""
    if not hasattr(blockamr, "MLMG"):
        pytest.skip("blockamr.MLMG binding not available")
    N = 16
    geom, ba, dm = _make_mesh(N)
    alpha, fx, fy, fz = _helmholtz_coeffs(geom, ba, dm, N)

    info = blockamr.LPInfo()
    abec = blockamr.MLABecLaplacian(geom, ba, dm, info)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3, [blockamr.LinOpBCType.Periodic] * 3
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)
    abec.set_a_coeffs(0, _const_cell(ba, dm, 1.0))
    abec.set_b_coeffs(
        0,
        _const_face(geom, dm, 0, N, 1.0),
        _const_face(geom, dm, 1, N, 1.0),
        _const_face(geom, dm, 2, N, 1.0),
    )
    mlmg = blockamr.MLMG(abec)

    with pytest.raises(RuntimeError, match="precond_mlmg"):
        blockamr.FaceCoeffSolver(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=gko_executor("reference"), solver="ir",
            precond_mlmg=mlmg,
        )
