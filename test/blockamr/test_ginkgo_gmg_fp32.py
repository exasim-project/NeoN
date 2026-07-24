# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""M5 FP32 V-cycle for the native GMG preconditioner (gmg_precision="fp32").

The whole V-cycle hierarchy (level coefficients, sol/rhs work fields, smoother,
residual / restriction / prolongation, ghost fills) runs in single precision
while the outer CG and matrix-free operator stay double: the FP64 residual norm
is the convergence authority. The FP32 V-cycle is still a fixed linear operator
to working precision, so CG tolerates it — its iteration count grows by at most
+2 vs the FP64 preconditioner, and the converged answer matches to solver
tolerance. Model problem is the same periodic Helmholtz as
``test_ginkgo_gmg_knobs.py``; gating/style mirror that file.
"""

import numpy as np
import pytest

import blockamr


def _make_mesh(n, periodic=True):
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    per = [1, 1, 1] if periodic else [0, 0, 0]
    geom = blockamr.Geometry(box, rb, 0, per)
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
    sol.set_val(0.0)
    return sol


def _max_abs_diff(a, b):
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


# ---------------------------------------------------------------------------
# FP32 V-cycle: convergence + iteration-count budget vs FP64
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("smoother", ["rbgs", "chebyshev"])
def test_gmg_fp32_within_two_iters_of_fp64(blockamr_session, executor, smoother):
    """FP32 V-cycle converges (FP64 residual) within +2 CG iters of the FP64 one."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s64 = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=200,
        rtol=1e-10,
        precond="gmg",
        gmg_smoother=smoother,
        gmg_precision="fp64",
    )
    st64 = s64.solve(rhs, _zero_sol(ba, dm))
    assert st64["converged"] is True

    s32 = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=200,
        rtol=1e-10,
        precond="gmg",
        gmg_smoother=smoother,
        gmg_precision="fp32",
    )
    st32 = s32.solve(rhs, _zero_sol(ba, dm))
    # The FP64 residual norm (res_norm is recomputed in double) is the authority.
    assert st32["converged"] is True
    assert st32["res_norm"] < 1e-6
    assert st32["num_iters"] <= st64["num_iters"] + 2, (
        f"fp32 {smoother} {st32['num_iters']} iters exceeds fp64 "
        f"{st64['num_iters']} + 2"
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_fp32_solution_matches_fp64(blockamr_session, executor):
    """The precond only steers CG: the converged fp32-precond answer matches fp64."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s64 = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=200, rtol=1e-10, precond="gmg"
    )
    sol64 = _zero_sol(ba, dm)
    s64.solve(rhs, sol64)

    s32 = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=200,
        rtol=1e-10,
        precond="gmg",
        gmg_precision="fp32",
    )
    sol32 = _zero_sol(ba, dm)
    st32 = s32.solve(rhs, sol32)
    assert st32["converged"] is True
    max_diff = _max_abs_diff(sol32, sol64)
    assert max_diff < 1e-6, f"Max |sol_fp32 - sol_fp64| = {max_diff} exceeds 1e-6"


def test_gmg_fp32_unknown_precision_raises(blockamr_session):
    """An unknown gmg_precision is rejected at construction."""
    N = 8
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    with pytest.raises(RuntimeError, match="unknown gmg_precision"):
        _make_solver_or_skip(
            coeffs, geom, "reference", solver="cg", precond="gmg", gmg_precision="fp16"
        )


# ---------------------------------------------------------------------------
# GmgConfig precision field (pure Python)
# ---------------------------------------------------------------------------
def test_gmg_config_precision_default_and_kwargs():
    """GmgConfig.precision defaults to fp64 and maps to the gmg_precision kwarg."""
    cfg = blockamr.GmgConfig()
    assert cfg.precision == "fp64"
    assert cfg.kwargs()["gmg_precision"] == "fp64"
    cfg32 = blockamr.GmgConfig(precision="fp32")
    assert cfg32.kwargs()["gmg_precision"] == "fp32"


def test_gmg_config_precision_validation():
    """An out-of-domain precision is rejected by pydantic."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        blockamr.GmgConfig(precision="fp16")


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_config_fp32_drives_solver(blockamr_session, executor):
    """A GmgConfig(precision="fp32") splatted into FaceCoeffSolver builds and solves."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    cfg = blockamr.GmgConfig(smoother="chebyshev", precision="fp32")
    s = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=100,
        rtol=1e-10,
        precond="gmg",
        **cfg.kwargs(),
    )
    stats = s.solve(rhs, _zero_sol(ba, dm))
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6
