# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""M4 V-cycle knobs for the native GMG preconditioner (precond="gmg").

Covers the ctor kwargs added in M4 — ``gmg_pre_sweeps`` / ``gmg_post_sweeps`` /
``gmg_coarsest_sweeps``, ``gmg_max_levels`` / ``gmg_min_bottom`` (hierarchy
truncation) and ``gmg_smoother="rbgs"|"chebyshev"`` — plus the pure-Python
``GmgConfig`` surface. Defaults reproduce the previous fixed behaviour (see
``test_ginkgo_gmg.py``), so those tests still pin the baseline. Model problem is
the same periodic Helmholtz with a seeded random rhs.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor


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


# ---------------------------------------------------------------------------
# Sweep-count knobs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_one_plus_one_sweeps_converges(blockamr_session, executor):
    """A symmetric 1+1 RB-GS V-cycle still converges (fewer/cheaper sweeps)."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=200,
        rtol=1e-10,
        precond="gmg",
        gmg_pre_sweeps=1,
        gmg_post_sweeps=1,
    )
    stats = s.solve(rhs, _zero_sol(ba, dm))
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6
    # 1+1 is a weaker smoother than the 2+2 default, so it needs >= as many iters.
    assert stats["num_iters"] <= 40, f"1+1 sweeps count unexpectedly high: {stats['num_iters']}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_asymmetric_sweeps_warns_but_runs(blockamr_session, executor, capfd):
    """pre != post is allowed but warns (non-symmetric V-cycle); it still solves.

    omega is pinned to 1.0 so this measures the axis it names. The default 1.1 is
    the OTHER symmetry breaker, and the two do not compose -- see
    ``test_asymmetric_sweeps_and_over_relaxation_stack`` below.
    """
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=300,
        rtol=1e-10,
        precond="gmg",
        gmg_pre_sweeps=2,
        gmg_post_sweeps=1,
        gmg_omega=1.0,
    )
    out = capfd.readouterr()
    assert "non-symmetric" in (out.err + out.out), "expected a symmetry warning on pre != post"
    stats = s.solve(rhs, _zero_sol(ba, dm))
    # Non-symmetric preconditioner: CG is not guaranteed, but for this mild
    # asymmetry it still reaches tolerance (just more iterations).
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6


def test_asymmetric_sweeps_and_over_relaxation_stack(blockamr_session):
    """The V-cycle's two symmetry breakers COMPOSE, and the pair is not survivable.

    Either alone is fine for CG: pre != post costs iterations but converges (the
    test above), and omega=1.1 with pre == post is the shipped default and the
    fastest configuration measured. Together they are not -- CG stops converging
    at all. That is the whole reason the default omega is safe to raise: the
    default sweeps are symmetric, and the asymmetric case has warned since it was
    added.

    Measured on this problem (N=32, precond="gmg", 300-iteration budget):

        sweeps    omega=1.0   omega=1.05   omega=1.1   omega=1.15
        2 / 1      16 iters    21 iters     diverges    diverges
        2 / 2       8 iters     8 iters      8 iters     8 iters
    """
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    def solve(pre, post, omega):
        s = _make_solver_or_skip(
            coeffs, geom, "cuda", solver="cg", max_iter=300, rtol=1e-10,
            precond="gmg", gmg_pre_sweeps=pre, gmg_post_sweeps=post, gmg_omega=omega,
        )
        return s.solve(rhs, _zero_sol(ba, dm))

    # Symmetric sweeps: over-relaxation is safe, which is what makes it the default.
    assert solve(2, 2, 1.1)["converged"] is True
    # Asymmetric sweeps alone: converges, just slower.
    assert solve(2, 1, 1.0)["converged"] is True
    # Both at once: does not.
    assert solve(2, 1, 1.1)["converged"] is False


# ---------------------------------------------------------------------------
# Chebyshev smoother
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_chebyshev_matches_rbgs(blockamr_session, executor):
    """Chebyshev-smoothed CG reaches rtol within +3 iterations of RB-GS."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s_rbgs = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=100, rtol=1e-10, precond="gmg"
    )
    st_rbgs = s_rbgs.solve(rhs, _zero_sol(ba, dm))
    assert st_rbgs["converged"] is True

    s_cheb = _make_solver_or_skip(
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=100,
        rtol=1e-10,
        precond="gmg",
        gmg_smoother="chebyshev",
    )
    st_cheb = s_cheb.solve(rhs, _zero_sol(ba, dm))
    assert st_cheb["converged"] is True
    assert st_cheb["res_norm"] < 1e-6
    assert st_cheb["num_iters"] <= st_rbgs["num_iters"] + 3, (
        f"chebyshev {st_cheb['num_iters']} not within +3 of rbgs {st_rbgs['num_iters']}"
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_chebyshev_flat_in_n(blockamr_session, executor):
    """The Chebyshev V-cycle keeps the CG count flat in N (still a real MG)."""
    iters = {}
    for n in (32, 64):
        geom, ba, dm = _make_mesh(n)
        coeffs = _helmholtz_coeffs(geom, ba, dm, n)
        rhs = _random_rhs(ba, dm)
        s = _make_solver_or_skip(
            coeffs,
            geom,
            executor,
            solver="cg",
            max_iter=100,
            rtol=1e-10,
            precond="gmg",
            gmg_smoother="chebyshev",
        )
        stats = s.solve(rhs, _zero_sol(ba, dm))
        assert stats["converged"] is True
        iters[n] = stats["num_iters"]
    assert iters[64] <= iters[32] + 5, f"chebyshev count grows with N: {iters}"
    assert iters[32] <= 25 and iters[64] <= 25, f"chebyshev count not small: {iters}"


def test_gmg_unknown_smoother_raises(blockamr_session):
    """An unknown gmg_smoother is rejected at construction."""
    N = 8
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    with pytest.raises(RuntimeError, match="unknown gmg_smoother"):
        _make_solver_or_skip(
            coeffs, geom, "reference", solver="cg", precond="gmg", gmg_smoother="bogus"
        )


# ---------------------------------------------------------------------------
# Hierarchy truncation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("kw", [{"gmg_max_levels": 2}, {"gmg_min_bottom": 8}])
def test_gmg_truncated_hierarchy_converges(blockamr_session, executor, kw):
    """Truncating the hierarchy (fewer levels / larger bottom) still converges."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=200, rtol=1e-10, precond="gmg", **kw
    )
    stats = s.solve(rhs, _zero_sol(ba, dm))
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6


# ---------------------------------------------------------------------------
# GmgConfig (pure Python)
# ---------------------------------------------------------------------------
def test_gmg_config_defaults_and_kwargs():
    """GmgConfig defaults map to the gmg_* / precond_cycles ctor kwargs."""
    cfg = blockamr.GmgConfig()
    assert cfg.kwargs() == {
        "gmg_pre_sweeps": 2,
        "gmg_post_sweeps": 2,
        "gmg_coarsest_sweeps": 16,
        "gmg_max_levels": 0,
        "gmg_min_bottom": 2,
        "gmg_omega": 1.1,
        "gmg_smoother": "rbgs",
        "gmg_precision": "fp64",
        "gmg_coeff_precision": "",
        "precond_cycles": 1,
    }
    cfg2 = blockamr.GmgConfig(smoother="chebyshev", pre_sweeps=3, post_sweeps=3, cycles=2)
    kw = cfg2.kwargs()
    assert kw["gmg_smoother"] == "chebyshev"
    assert kw["gmg_pre_sweeps"] == 3 and kw["gmg_post_sweeps"] == 3
    assert kw["precond_cycles"] == 2


def test_gmg_config_validation():
    """Out-of-range / invalid GmgConfig fields are rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        blockamr.GmgConfig(pre_sweeps=-1)
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(coarsest_sweeps=0)
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(cycles=0)
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(min_bottom=1)
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(smoother="bogus")
    # omega must stay inside (0, 2) to be a convergent relaxation; the bounds are
    # exclusive, so both endpoints are rejected.
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(omega=0.0)
    with pytest.raises(ValidationError):
        blockamr.GmgConfig(omega=2.0)


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_config_drives_solver(blockamr_session, executor):
    """A GmgConfig splatted into FaceCoeffSolver builds and solves."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)
    cfg = blockamr.GmgConfig(smoother="chebyshev")
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
