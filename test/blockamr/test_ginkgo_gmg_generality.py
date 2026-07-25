# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Generality of the native GMG path along the axes MLMG covers and we did not test.

Every other ``test_ginkgo_*`` file poses the same problem shape: ONE box, a cubic
``[0,1]^3`` domain, CONSTANT face coefficients, periodic BCs and a power-of-two N.
That case is spectrally trivial (constant coefficients + periodic BCs make the
operator diagonal in Fourier space), so it says nothing about the situations MLMG is
built for. This file covers the axes that were untested:

  1. multi-box decompositions (``max_grid_size < N``) — MLMG agglomerates and
     consolidates coarse levels; we do not, so our hierarchy stops as soon as a
     BOX is no longer coarsenable;
  2. spatially varying coefficients, smooth and with a 1e4 jump (MLMG's ``b``);
  3. anisotropic cells (MLMG has semicoarsening; we always coarsen all 3 axes);
  4. non-power-of-two / unequal extents;
  5. a singular (all-periodic, alpha=0) system under the V-cycle;
  6. MLMG's own bottom solver, as the reference point for ours;
  7. inhomogeneous Dirichlet data, which we have no API for, via the rhs fold;
  8. multi-rank MPI;
  9. multi-component (``ncomp > 1``) fields;
 10. composite (multi-level AMR) systems under the native V-cycle;
 11. Robin BCs.

These are ACCURACY tests, not benchmarks: the meshes are 8^3-18^3 so the whole file
runs in a couple of seconds, and nothing asserts an iteration count except
``test_mlmg_bottom_solver_costs_cycles``, whose entire subject is a cycle count.

The measured verdict, which the tests below pin: the PRECONDITIONED path
(``solver="cg", precond="gmg"``) is accurate on every axis — Krylov absorbs a weak
V-cycle — while the stationary path (``solver="gmg"``) is not, and either stalls or
diverges outright. Those stationary failures are recorded as ``xfail``, so they flip
to XPASS the day the underlying gap (agglomeration, semicoarsening) is closed.

Features that are out of scope rather than broken are still written out, so that
removing a marker is all it takes when they land. AMR gets an ``xfail`` rather than a
``skip`` on purpose: its body solves a real 2-level hierarchy against MLMG on every
run and only the preconditioner assertions fail, so the multi-level operator is
genuinely exercised. MPI (needs ``mpirun``) and Robin BCs (no ghost fill for it) have
nothing runnable behind them and stay ``skip``.

Referees: MLMG where it can solve the problem, the manufactured solution where there
is one, and otherwise an independent numpy residual of the same 7-point operator —
MLMG aborts the process (``MLMG failing so lets stop here``) on the anisotropic and
high-contrast cases, so it cannot referee those.
"""

import os

import numpy as np
import pytest

import blockamr

# Accuracy target for every solve here; also the agreement threshold against a
# referee, one decade looser to leave room for a different iteration order.
RTOL = 1e-11
AGREE = 1e-9

# The stationary V-cycle needs a strong bottom solve to be a solver at all (the
# default coarsest_sweeps=16 is tuned for PRECONDITIONED CG, where Krylov absorbs
# a merely-decent bottom, and still leaves the stationary path bottom-limited even
# single-box). These knobs make it converge in ~10 cycles on the easy problem, so a
# failure in the tests below is attributable to the axis under test and not to a
# weak bottom.
STRONG_BOTTOM = {"gmg_coarsest_sweeps": 64, "gmg_min_bottom": 2}

# Cycle budget for the stationary solver. Not a performance bound: 10 cycles solve
# the easy problem at every size used here, so "does not finish in 30" means the
# V-cycle has lost its grid-independent convergence, which is a correctness claim.
STATIONARY_BUDGET = 30

PRECONDITIONED = {"solver": "cg", "precond": "gmg", "max_iter": 400, "rtol": RTOL}
STATIONARY = {"solver": "gmg", "max_iter": STATIONARY_BUDGET, "rtol": RTOL, **STRONG_BOTTOM}

# Every accuracy test runs single-box AND decomposed, because a box boundary is a
# different code path in three places: FillBoundary supplies the stencil's ghosts
# instead of the periodic wrap, the RB-GS smoother has to refresh ghosts between
# COLOUR sweeps (gmg_precond.hpp rbgsSmooth), and the coarse hierarchy truncates
# when a box stops being coarsenable. _run refuses a max_size that does not actually
# split the domain, so a "multibox" case cannot silently re-run the single-box one.
MULTIBOX = [False, True]
MULTIBOX_IDS = ["1box", "multibox"]


def _split(shape, multibox):
    """max_size that halves the shortest extent (so every axis is cut), or None."""
    return min(shape) // 2 if multibox else None


# ---------------------------------------------------------------------------
# mesh / field helpers
# ---------------------------------------------------------------------------
def _make_mesh(shape, max_size=None, prob_hi=(1.0, 1.0, 1.0), periodic=True):
    """Mesh on [0, prob_hi] with shape cells, cut into boxes of at most max_size."""
    box = blockamr.Box([0, 0, 0], [shape[0] - 1, shape[1] - 1, shape[2] - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], list(prob_hi))
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1] if periodic else [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(shape) if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _face_ba(ba, d):
    """Face BoxArray in direction d, box-for-box with the cell BoxArray.

    convert_ba keeps the box order, so the cell DistributionMapping still applies —
    which is what lets these tests use more than one box (surrounding_nodes on the
    whole domain box, as the other test files do, only matches for a single box).
    """
    typ = [0, 0, 0]
    typ[d] = 1
    return blockamr.convert_ba(ba, blockamr.IntVect(*typ))


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _scatter(mf, values):
    """Fill the valid region of a (possibly multi-box) MultiFab from a global array."""
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1]
        mf.copy_from(mfi, arr)
    return mf


def _gather(mf, shape, comp=0):
    """Assemble one component of a (possibly multi-box) MultiFab into one array.

    Single-rank only — under MPI a rank sees only its own boxes, so the MPI test
    sums the per-rank arrays instead.
    """
    out = np.full(shape, np.nan)
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        out[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1] = arr[:, :, :, comp]
    assert not np.isnan(out).any(), "gather missed cells — face/cell BoxArrays disagree"
    return out


def _cell_mf(ba, dm, values):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return _scatter(mf, values)


def _face_mf(ba, dm, d, values):
    return _scatter(blockamr.MultiFab(_face_ba(ba, d), dm, 1, 0), values)


def _zero_sol(ba, dm, ncomp=1):
    sol = blockamr.MultiFab(ba, dm, ncomp, 1)
    sol.set_val(0.0)
    return sol


def _random_values(shape, seed=42, meanzero=False):
    """Seeded random cell values — full spectrum, so the solver must genuinely work."""
    v = np.random.default_rng(seed).standard_normal(shape)
    return v - v.mean() if meanzero else v


def _cell_centres(shape, dx):
    return np.meshgrid(*[(np.arange(shape[d]) + 0.5) * dx[d] for d in range(3)], indexing="ij")


def _nboxes(mf):
    """Number of boxes this rank owns — single rank, so the whole BoxArray."""
    return sum(1 for _ in blockamr.MFIterator(mf))


# ---------------------------------------------------------------------------
# coefficients and referees
# ---------------------------------------------------------------------------
def _b_on_faces(shape, dx, d, bfn):
    """b sampled where direction d's faces live: face-centred in d, cell-centred else.

    On a periodic axis the wrap face (index n in direction d, at prob_hi) is the SAME
    physical face as index 0, so bfn must give it the same value — otherwise the
    matrix is not symmetric and CG diverges rather than converging slowly.
    """
    ns = list(shape)
    ns[d] += 1
    axes = [(np.arange(ns[a]) + (0.0 if a == d else 0.5)) * dx[a] for a in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return bfn(x, y, z)


def _coeffs(geom, ba, dm, shape, bfn=None, a_val=1.0):
    """(solver args, MLMG b coeffs, numpy referee data) for a*u - div(b grad u).

    Face-coefficient form of MLABecLaplacian: cell source alpha = a_val, face coeff
    -b/dx_d^2. The SAME MultiFab serves as ux and lx because the stencil reads
    ux(i+1) and lx(i) — a single face field is therefore already symmetric.
    """
    dx = geom.cell_size()
    if bfn is None:

        def bfn(x, y, z):
            return np.ones_like(x)

    b = [_b_on_faces(shape, dx, d, bfn) for d in range(3)]
    f = [-b[d] / dx[d] ** 2 for d in range(3)]
    alpha = _const_cell(ba, dm, a_val)
    fx, fy, fz = (_face_mf(ba, dm, d, f[d]) for d in range(3))
    bx, by, bz = (_face_mf(ba, dm, d, b[d]) for d in range(3))
    return (alpha, fx, fx, fy, fy, fz, fz), (bx, by, bz), (a_val, f)


def _rel_residual(sol, rhs, referee):
    """||rhs - A sol||_2 / ||rhs||_2 with A rebuilt in numpy (independent of the C++).

    Mirrors the kernel exactly: aE = ux(i+1), aW = lx(i), ...,
    diag = alpha - sum(a), A u = diag*u + sum(a * u_neighbour), periodic wrap.
    """
    a_val, f = referee
    upper = [f[0][1:, :, :], f[1][:, 1:, :], f[2][:, :, 1:]]
    lower = [f[0][:-1, :, :], f[1][:, :-1, :], f[2][:, :, :-1]]
    diag = a_val - sum(upper[d] + lower[d] for d in range(3))
    au = diag * sol
    for d in range(3):
        au = au + upper[d] * np.roll(sol, -1, axis=d) + lower[d] * np.roll(sol, 1, axis=d)
    return float(np.linalg.norm(rhs - au) / np.linalg.norm(rhs))


def _make_solver_or_skip(coeffs, geom, executor, **kwargs):
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")
    try:
        return blockamr.FaceCoeffSolver(*coeffs, geom, executor=executor, **kwargs)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise


def _run(
    shape,
    executor,
    cfg,
    *,
    values=None,
    rhs_extra=None,
    bfn=None,
    a_val=1.0,
    max_size=None,
    prob_hi=(1.0, 1.0, 1.0),
    periodic=True,
    seed=42,
    meanzero=False,
    **solver_kw,
):
    """Build the problem, solve it, and return (stats, solution, rhs values, referee).

    Every AMReX and Ginkgo object stays LOCAL to this function on purpose. A failing
    assertion in a test body keeps that frame's locals alive for the traceback until
    the session ends — i.e. past the blockamr_session teardown that finalizes AMReX —
    and freeing device memory after the CUDA context is gone aborts the interpreter
    with `CUDA error 709: context is destroyed`. Returning plain numpy keeps the
    xfail bodies below harmless.

    `rhs_extra` is added to the rhs after the referee's copy of `values` is taken, so
    a boundary-condition fold does not corrupt the residual check.

    Adds `nboxes` to the returned stats, and refuses a `max_size` that does not
    actually split the domain — otherwise a multi-box parametrisation would quietly
    re-run the single-box case and pass for the wrong reason.
    """
    geom, ba, dm = _make_mesh(shape, max_size=max_size, prob_hi=prob_hi, periodic=periodic)
    coeffs, _, referee = _coeffs(geom, ba, dm, shape, bfn=bfn, a_val=a_val)
    nboxes = _nboxes(coeffs[0])
    if max_size is not None:
        assert nboxes > 1, f"max_size={max_size} did not split {shape} — test is vacuous"
    if values is None:
        values = _random_values(shape, seed=seed, meanzero=meanzero)
    rhs = _cell_mf(ba, dm, values if rhs_extra is None else values + rhs_extra)
    solver = _make_solver_or_skip(coeffs, geom, executor, **cfg, **solver_kw)
    sol = _zero_sol(ba, dm)
    stats = dict(solver.solve(rhs, sol))
    stats["nboxes"] = nboxes
    return stats, _gather(sol, shape), values, referee


def _mlmg_run(shape, bfn=None, a_val=1.0, bottom=None, values=None, seed=42, meanzero=False):
    """Reference MLMG solve of the same operator; returns (solution array, cycles).

    Device objects stay local here for the same reason as in _run.
    """
    if not hasattr(blockamr, "MLABecLaplacian"):
        pytest.skip("blockamr.MLABecLaplacian binding not available")
    geom, ba, dm = _make_mesh(shape)
    _, bcoeff, _ = _coeffs(geom, ba, dm, shape, bfn=bfn, a_val=a_val)
    if values is None:
        values = _random_values(shape, seed=seed, meanzero=meanzero)
    rhs = _cell_mf(ba, dm, values)

    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    per = [blockamr.LinOpBCType.Periodic] * 3
    abec.set_domain_bc(per, per)
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)  # alpha_scalar, beta_scalar
    abec.set_a_coeffs(0, _const_cell(ba, dm, a_val))
    abec.set_b_coeffs(0, *bcoeff)
    sol = _zero_sol(ba, dm)
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    if bottom is not None:
        mlmg.set_bottom_solver(bottom)
    mlmg.solve(sol, rhs, RTOL, 1e-14)
    return _gather(sol, shape), mlmg.get_num_iters()


# ---------------------------------------------------------------------------
# 1. multi-box decompositions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_preconditioned_is_decomposition_independent(blockamr_session, executor):
    """cg+gmg gives the same answer for 1, 8, 64 and 512 boxes.

    Splitting a periodic constant-coefficient domain into boxes does not change the
    matrix (ghosts come from FillBoundary), so every decomposition must give the same
    solution — even though the coarse hierarchy DOES change, because coarsening stops
    when a box is no longer coarsenable. The 2-cell boxes at max_size=2 are the
    extreme: the hierarchy is one level deep, so this also pins that a V-cycle which
    cannot coarsen at all is still a valid (if useless) preconditioner.
    """
    shape = (16, 16, 16)
    values = _random_values(shape)
    ref = None
    seen = {}
    for max_size in (None, 8, 4, 2):
        stats, sol, _, _ = _run(shape, executor, PRECONDITIONED, values=values, max_size=max_size)
        assert stats["converged"] is True, f"max_size={max_size} did not converge"
        seen[max_size] = stats["nboxes"]
        if ref is None:
            ref = sol
        diff = float(np.max(np.abs(sol - ref)))
        assert diff < AGREE, f"max_size={max_size} ({stats['nboxes']} boxes) differs by {diff}"
    assert sorted(seen.values()) == [1, 8, 64, 512], f"unexpected decompositions: {seen}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.xfail(
    reason="no coarse-grid agglomeration: coarsening stops when a BOX is no longer "
    "coarsenable, so many small boxes leave a large bottom grid that the "
    "smoothing-only bottom solve cannot handle. Measured 10 -> 66 cycles at N=16 "
    "for 1 -> 64 boxes, and no convergence at all for 512 boxes at N=64. MLMG "
    "agglomerates and consolidates instead.",
    strict=False,
)
def test_native_gmg_is_decomposition_independent(blockamr_session, executor):
    """The stationary V-cycle should also be decomposition-independent. It is not."""
    shape = (16, 16, 16)
    values = _random_values(shape)
    stats1, sol1, _, referee = _run(shape, executor, STATIONARY, values=values)
    stats64, sol64, _, _ = _run(shape, executor, STATIONARY, values=values, max_size=4)
    assert stats1["converged"] is True, "single box did not converge"
    assert stats64["converged"] is True, "64 boxes did not converge"
    assert _rel_residual(sol64, values, referee) < AGREE
    assert float(np.max(np.abs(sol1 - sol64))) < AGREE


# ---------------------------------------------------------------------------
# 2. variable coefficients
# ---------------------------------------------------------------------------
def _smooth_b(x, y, z):
    """Smoothly varying, periodic on [0,1] so the wrap face stays consistent."""
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)


def _jump_b(x, y, z, contrast=1.0e4):
    """Interior slab of high diffusivity; b = 1 on both wrap faces (x=0 and x=1)."""
    return np.where((x > 0.25) & (x < 0.75), contrast, 1.0)


@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_smooth_variable_coeffs_matches_mlmg(blockamr_session, executor, multibox):
    """A smoothly varying b is rediscretised correctly: cg+gmg matches MLMG.

    The hierarchy coarsens b by face-averaging; this is the first test that puts a
    non-constant coefficient through it. The problem is also no longer spectrally
    trivial — unpreconditioned CG needs several times as many iterations here as on
    the constant-coefficient problem of the same size. The MLMG reference stays
    single-box on purpose, so the multi-box run is checked against an independent
    decomposition as well as an independent implementation.
    """
    shape = (16, 16, 16)
    ref, mlmg_cycles = _mlmg_run(shape, bfn=_smooth_b)
    assert mlmg_cycles > 1
    stats, sol, _, _ = _run(
        shape, executor, PRECONDITIONED, bfn=_smooth_b, max_size=_split(shape, multibox)
    )
    assert stats["converged"] is True
    diff = float(np.max(np.abs(sol - ref)))
    assert diff < 1e-6, f"max |sol - sol_mlmg| = {diff} exceeds 1e-6"


@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("smoother", ["rbgs", "chebyshev"])
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_coefficient_jump_robust(blockamr_session, executor, smoother, multibox):
    """cg+gmg solves a 1e4 coefficient jump, checked by an independent numpy residual.

    MLMG cannot referee this one — it aborts the process on this problem — so the
    residual of the converged solution is recomputed in numpy from the same face
    coefficients. Decomposed, the jump at x=0.25/0.75 lands inside a box on one side
    and on a box boundary on the other, so the coefficient discontinuity is crossed
    by both the stencil and the ghost exchange.
    """
    shape = (16, 16, 16)
    stats, sol, values, referee = _run(
        shape,
        executor,
        PRECONDITIONED,
        bfn=_jump_b,
        gmg_smoother=smoother,
        max_size=_split(shape, multibox),
    )
    assert stats["converged"] is True
    res = _rel_residual(sol, values, referee)
    assert res < AGREE, f"converged flag but an independent residual of {res}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.xfail(
    reason="the stationary V-cycle is not robust to coefficient jumps: at 1e4 contrast "
    "it does not converge at all (400 cycles reach a relative residual of 1.2e-2 "
    "at N=32), and at 1e2 it needs 203 cycles. CG with the SAME V-cycle takes "
    "14-16 iterations at either contrast.",
    strict=False,
)
def test_native_gmg_coefficient_jump(blockamr_session, executor):
    """The stationary V-cycle should handle a coefficient jump. It does not."""
    shape = (16, 16, 16)
    stats, sol, values, referee = _run(shape, executor, STATIONARY, bfn=_jump_b)
    assert stats["converged"] is True
    assert _rel_residual(sol, values, referee) < AGREE


# ---------------------------------------------------------------------------
# 3. MLMG's bottom solver — the reference point for our cycle counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [16, 32])
def test_mlmg_bottom_solver_costs_cycles(blockamr_session, n):
    """MLMG needs ~2x the cycles when its bottom solver is smoothing-only, like ours.

    The one deliberate cycle-count assertion in this file: it is the apples-to-apples
    reference for our own counts. Our bottom is `coarsest_sweeps` smoothing sweeps,
    i.e. BottomSolver::smoother, where MLMG defaults to BiCGStab down there. Measured
    9 -> 20 cycles at N=32 and 9 -> 17 at N=64, which accounts for our cycle count
    sitting above MLMG's on the same hierarchy and says the gap is the bottom solve
    rather than the smoother.
    """
    shape = (n, n, n)
    _, krylov = _mlmg_run(shape, bottom="bicgstab")
    _, smoother = _mlmg_run(shape, bottom="smoother")
    assert smoother >= 1.5 * krylov, (
        f"expected a smoothing-only bottom to cost cycles: bicgstab={krylov} smoother={smoother}"
    )


# ---------------------------------------------------------------------------
# 4. anisotropic cells
# ---------------------------------------------------------------------------
ANISOTROPIC = {"prob_hi": (1.0, 1.0, 4.0)}  # cubic index space -> dz = 4 dx


@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_anisotropic_cells(blockamr_session, executor, multibox):
    """cg+gmg stays accurate on dz = 4 dx.

    We coarsen all three directions unconditionally; MLMG has semicoarsening for
    this, so the V-cycle is a weak preconditioner here — but CG still delivers the
    solution. (MLMG itself aborts on this problem without semicoarsening, so it
    cannot referee it; the referee is the numpy residual.)
    """
    shape = (16, 16, 16)
    stats, sol, values, referee = _run(
        shape, executor, PRECONDITIONED, max_size=_split(shape, multibox), **ANISOTROPIC
    )
    assert stats["converged"] is True
    res = _rel_residual(sol, values, referee)
    assert res < AGREE, f"converged flag but an independent residual of {res}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.xfail(
    reason="the stationary V-cycle DIVERGES on anisotropic cells: measured a residual "
    "of 8.6e+63 after 200 cycles at dz/dx=4, N=16, because coarsening all three "
    "directions leaves modes the smoother cannot damp. MLMG uses semicoarsening, "
    "which we do not expose. CG with the same V-cycle converges in 29 iterations.",
    strict=False,
)
def test_native_gmg_anisotropic_cells(blockamr_session, executor):
    """The stationary V-cycle should handle anisotropic cells. It diverges instead."""
    shape = (16, 16, 16)
    stats, sol, values, referee = _run(shape, executor, STATIONARY, **ANISOTROPIC)
    assert stats["converged"] is True
    assert _rel_residual(sol, values, referee) < AGREE


# ---------------------------------------------------------------------------
# 5. non-power-of-two / unequal extents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("shape", [(12, 16, 20), (18, 18, 18)])
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_odd_extents(blockamr_session, executor, shape, multibox):
    """cg+gmg handles unequal and non-power-of-two extents.

    18^3 is the hard one: 18 -> 9 and 9 is odd, so the hierarchy is only two levels
    deep and the V-cycle is a poor preconditioner — CG still delivers the solution.

    The decompositions are the interesting part here. (12,16,20) at max_size=6 gives
    24 boxes of FOUR different sizes — (6,4,4), (6,4,6), (6,6,4), (6,6,6) — so the
    face BoxArrays, the ghost exchange and the flat gather/scatter order all have to
    cope with unequal boxes. 18^3 at max_size=9 gives 9^3 boxes, which are not
    coarsenable at all, so the hierarchy collapses to a single level and the
    "V-cycle" degenerates into plain smoothing; CG has to carry it alone.
    """
    stats, sol, values, referee = _run(
        shape, executor, PRECONDITIONED, max_size=_split(shape, multibox)
    )
    assert stats["converged"] is True
    res = _rel_residual(sol, values, referee)
    assert res < AGREE, f"converged flag but an independent residual of {res}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.xfail(
    reason="the stationary V-cycle loses its grid-independence when the extents stop "
    "the hierarchy early: 18 -> 9 is not coarsenable, leaving 2 levels, so it "
    "needs 90 cycles at 18^3 where 16^3 needs 10 (238 vs 10 at 30^3). It does "
    "converge eventually; CG with the same V-cycle needs 12 iterations.",
    strict=False,
)
def test_native_gmg_odd_extents(blockamr_session, executor):
    """The stationary V-cycle should stay grid-independent at 18^3. It needs 9x the cycles."""
    stats, sol, values, referee = _run((18, 18, 18), executor, STATIONARY)
    assert stats["converged"] is True
    assert _rel_residual(sol, values, referee) < AGREE


# ---------------------------------------------------------------------------
# 6. singular system under the V-cycle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("cfg", [PRECONDITIONED, STATIONARY], ids=["cg+gmg", "native"])
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_singular_projected_matches_mlmg(blockamr_session, executor, cfg, multibox):
    """alpha=0 periodic Poisson (constants in the nullspace) under the V-cycle.

    ``project_nullspace`` projects the outer rhs and iterate mean-zero; the coarse
    levels are NOT projected, so this test exists to show that the V-cycle does not
    accumulate a constant regardless — for the preconditioned AND the stationary path.
    """
    shape = (16, 16, 16)
    values = _random_values(shape, meanzero=True)
    ref, _ = _mlmg_run(shape, a_val=0.0, values=values)
    ref = ref - ref.mean()

    stats, sol, _, _ = _run(shape, executor, cfg, a_val=0.0, values=values, project_nullspace=True)
    assert stats["converged"] is True
    assert abs(float(sol.mean())) < 1e-10, f"solution mean {sol.mean()} not ~0"
    diff = float(np.max(np.abs((sol - sol.mean()) - ref)))
    assert diff < 1e-6, f"max |sol - sol_mlmg| = {diff} exceeds 1e-6"


# ---------------------------------------------------------------------------
# 7. inhomogeneous Dirichlet data
# ---------------------------------------------------------------------------
def _dirichlet_fold(shape, dx, gfn):
    """rhs correction that turns the homogeneous-Dirichlet operator inhomogeneous.

    ``bc=["dirichlet", ...]`` is HOMOGENEOUS: the ghost fill is
    ``ghost = -interior``, i.e. u = 0 on the face. For u = g on the face the fill
    would be ``ghost = 2g - interior``, which differs by exactly ``2 g``, so

        A_inhom u = A_home u + 2 a_face g   ->   solve A_home u = f - 2 a_face g,

    summed over every domain face a cell touches (a corner cell gets three terms).
    """
    x, y, z = _cell_centres(shape, dx)
    fold = np.zeros(shape)
    for d in range(3):
        a_face = -1.0 / dx[d] ** 2
        for side, coord in ((0, 0.0), (-1, shape[d] * dx[d])):
            face = [x, y, z]
            face[d] = np.full_like(x, coord)
            sl = tuple(
                (slice(0, 1) if side == 0 else slice(-1, None)) if a == d else slice(None)
                for a in range(3)
            )
            fold[sl] += -2.0 * a_face * gfn(*face)[sl]
    return fold


@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_inhomogeneous_dirichlet_via_rhs_fold(blockamr_session, executor, multibox):
    """Inhomogeneous Dirichlet data has no API here, but folds exactly into the rhs.

    Pins the fold (see _dirichlet_fold) against the manufactured solution
    u = x^2+y^2+z^2 of u - lap u = u - 6, and pins its SECOND-ORDER accuracy: the
    fold itself is exact, and the linear ghost reconstruction of a quadratic is what
    leaves the O(dx^2) error. This is why an inhomogeneous-BC API would be
    convenience rather than capability.

    Decomposed, only the boxes that touch a domain face get the reflect fill
    (bc.hpp bcGhostFill returns false otherwise) while the interior boxes take their
    ghosts from FillBoundary — so the multibox case checks that the fold lands on
    exactly the cells whose ghosts are reflected.
    """

    def u_exact(x, y, z):
        return x**2 + y**2 + z**2

    errors = {}
    for n in (8, 16):
        shape = (n, n, n)
        dx = (1.0 / n,) * 3
        x, y, z = _cell_centres(shape, dx)
        exact = u_exact(x, y, z)
        stats, sol, _, _ = _run(
            shape,
            executor,
            PRECONDITIONED,
            values=exact - 6.0,  # (1 - lap)(x^2+y^2+z^2)
            rhs_extra=_dirichlet_fold(shape, dx, u_exact),
            periodic=False,
            bc=["dirichlet"] * 6,
            max_size=_split(shape, multibox),
        )
        assert stats["converged"] is True
        errors[n] = float(np.max(np.abs(sol - exact)))

    assert errors[16] < 2.0e-3, f"inhomogeneous Dirichlet fold is inaccurate: {errors}"
    assert errors[8] / errors[16] > 3.5, f"not second-order in dx: {errors}"


# ---------------------------------------------------------------------------
# 8. MPI — out of scope, body ready
# ---------------------------------------------------------------------------
def _n_ranks():
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(var)
        if value:
            return int(value)
    return 1


def _run_multirank(shape, cfg, values):
    """Solve, then assemble the solution across ranks (valid regions are disjoint)."""
    mpi = pytest.importorskip("mpi4py.MPI", reason="needs mpi4py for the cross-rank gather")
    geom, ba, dm = _make_mesh(shape, max_size=8)
    coeffs, _, referee = _coeffs(geom, ba, dm, shape)
    rhs = _cell_mf(ba, dm, values)
    solver = _make_solver_or_skip(coeffs, geom, "cuda", **cfg)
    sol = _zero_sol(ba, dm)
    stats = dict(solver.solve(rhs, sol))

    local = np.zeros(shape)
    for mfi in blockamr.MFIterator(sol):
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        arr = sol.copy_to_host(mfi)
        local[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = arr[:, :, :, 0]
    glob = np.zeros(shape)
    mpi.COMM_WORLD.Allreduce(local, glob, op=mpi.SUM)
    return stats, glob, referee


@pytest.mark.skipif(_n_ranks() < 2, reason="single rank; run under `mpirun -n 2 pytest`")
@pytest.mark.xfail(
    reason="MPI is out of scope and known-broken in two places: the native path's "
    "stopping test compares a RANK-LOCAL residual (gmg_kernels.hpp "
    "faceCoeffResidScatterNorm* reduce with ParReduce and no ParallelAllReduce) "
    "against a globally reduced ||rhs||, so it stops early; and the Ginkgo path's "
    "Dense vectors and their dot products are rank-local, so CG's scalars are "
    "wrong. Fix the norm first (see the rank-local-norm task).",
    strict=False,
)
@pytest.mark.parametrize("cfg", [PRECONDITIONED, STATIONARY], ids=["cg+gmg", "native"])
def test_gmg_multirank(blockamr_session, cfg):
    """The solve should give the same answer on 2 ranks as on 1."""
    shape = (16, 16, 16)
    values = _random_values(shape)
    stats, sol, referee = _run_multirank(shape, cfg, values)
    assert stats["converged"] is True
    assert _rel_residual(sol, values, referee) < AGREE


# ---------------------------------------------------------------------------
# 9. multi-component fields
# ---------------------------------------------------------------------------
def _run_multicomponent(shape, executor, cfg, comps, max_size=None):
    """Solve with an ncomp=len(comps) rhs/sol; returns (stats, [per-comp arrays], referee)."""
    geom, ba, dm = _make_mesh(shape, max_size=max_size)
    coeffs, _, referee = _coeffs(geom, ba, dm, shape)
    ncomp = len(comps)
    rhs = blockamr.MultiFab(ba, dm, ncomp, 0)
    rhs.set_val(0.0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        bx = mfi.valid_box()
        lo, hi = bx.small_end(), bx.big_end()
        for c, values in enumerate(comps):
            arr[:, :, :, c] = values[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
        rhs.copy_from(mfi, arr)
    solver = _make_solver_or_skip(coeffs, geom, executor, **cfg)
    sol = _zero_sol(ba, dm, ncomp=ncomp)
    stats = dict(solver.solve(rhs, sol))
    return stats, [_gather(sol, shape, comp=c) for c in range(ncomp)], referee


@pytest.mark.parametrize("multibox", MULTIBOX, ids=MULTIBOX_IDS)
@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.xfail(
    reason="ncomp > 1 is out of scope: a multi-component rhs/sol is accepted SILENTLY "
    "and only component 0 is solved (measured: component 1 of the solution is "
    "left untouched and its rhs ignored). MLMG solves every component. Either "
    "solve them all or reject the field — the silent reduction is the hazard.",
    strict=False,
)
def test_multicomponent_field(blockamr_session, executor, multibox):
    """A 2-component solve should solve both components (or refuse the field)."""
    shape = (8, 8, 8)
    comps = [_random_values(shape, seed=1), _random_values(shape, seed=2)]
    stats, sols, referee = _run_multicomponent(
        shape, executor, PRECONDITIONED, comps, max_size=_split(shape, multibox)
    )
    assert stats["converged"] is True
    for c, values in enumerate(comps):
        res = _rel_residual(sols[c], values, referee)
        assert res < AGREE, f"component {c} not solved: residual {res}"


# ---------------------------------------------------------------------------
# 10. composite (multi-level AMR) systems — out of scope
# ---------------------------------------------------------------------------
AMR_N = 8  # coarse cells per side
AMR_PATCH = (2, 5)  # coarse index range refined at ratio 2 -> fine box [4,11]^3


def _amr_levels():
    """(geom, ba, dm) per level for a 2-level periodic ratio-2 hierarchy."""
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    box_c = blockamr.Box([0, 0, 0], [AMR_N - 1] * 3)
    geom_c = blockamr.Geometry(box_c, rb, 0, [1, 1, 1])
    ba_c = blockamr.BoxArray(box_c)
    ba_c.max_size(AMR_N)

    box_f_dom = blockamr.Box([0, 0, 0], [2 * AMR_N - 1] * 3)
    geom_f = blockamr.Geometry(box_f_dom, rb, 0, [1, 1, 1])
    patch = blockamr.Box([2 * AMR_PATCH[0]] * 3, [2 * AMR_PATCH[1] + 1] * 3)
    ba_f = blockamr.BoxArray(patch)
    ba_f.max_size(2 * AMR_N)

    return [
        (geom_c, ba_c, blockamr.DistributionMapping(ba_c)),
        (geom_f, ba_f, blockamr.DistributionMapping(ba_f)),
    ]


def _amr_abec(levels):
    """Composite MLABecLaplacian (alpha=beta=1, unit coefficients, periodic)."""
    abec = blockamr.MLABecLaplacian(
        [lv[0] for lv in levels], [lv[1] for lv in levels], [lv[2] for lv in levels]
    )
    per = [blockamr.LinOpBCType.Periodic] * 3
    abec.set_domain_bc(per, per)
    abec.set_scalars(1.0, 1.0)
    for lev, (_geom, ba, dm) in enumerate(levels):
        abec.set_level_bc(lev, None)
        abec.set_a_coeffs(lev, _const_cell(ba, dm, 1.0))
        faces = []
        for d in range(3):
            mf = blockamr.MultiFab(_face_ba(ba, d), dm, 1, 0)
            mf.set_val(1.0)
            faces.append(mf)
        abec.set_b_coeffs(lev, *faces)
    return abec


def _amr_random(ba, dm, seed):
    """Random cell values on a level whose box need not start at index 0.

    The fine patch lives at [4,11]^3, so the global-array helpers (_scatter/_gather)
    do not apply — fill and compare per box instead.
    """
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(0.0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _amr_max_diff(a, b):
    """Max-norm difference between the valid regions of two same-BoxArray MultiFabs."""
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


def _run_composite_amr(executor, **extra):
    """Solve the 2-level AMR system matrix-free; return (stats, per-level max diff vs MLMG).

    Device objects stay local for the reason given in _run — and an unsupported
    keyword is CAUGHT here and reported as data rather than raised, so the xfail
    below fails in the test body (holding only dicts and floats) instead of unwinding
    through this frame with a hierarchy of MultiFabs alive in it.
    """
    if not hasattr(blockamr, "ginkgo_solve_composite"):
        pytest.skip("blockamr.ginkgo_solve_composite binding not available")
    levels = _amr_levels()
    rhs = [_amr_random(ba, dm, seed=11 + lev) for lev, (_g, ba, dm) in enumerate(levels)]

    # MLMG's own composite solve is the referee.
    ref = [_zero_sol(ba, dm) for _g, ba, dm in levels]
    mlmg = blockamr.MLMG(_amr_abec(levels))
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(ref, rhs, 1e-12, 0.0)

    got = [_zero_sol(ba, dm) for _g, ba, dm in levels]
    try:
        stats = dict(
            blockamr.ginkgo_solve_composite(
                _amr_abec(levels),
                got,
                rhs,
                executor=executor,
                max_iter=2000,
                rtol=1e-12,
                sign=+1.0,
                **extra,
            )
        )
    except TypeError as exc:  # keyword the binding does not have
        return {"unsupported": f"{exc}"}, []
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise

    # Covered coarse cells are slaved to the fine level, so average_down both before
    # comparing the coarse level (MLMG's convention).
    for pair in (ref, got):
        blockamr.average_down(
            pair[1], pair[0], levels[1][0], levels[0][0], 0, 1, blockamr.IntVect(2, 2, 2)
        )
    return stats, [_amr_max_diff(got[lev], ref[lev]) for lev in range(len(levels))]


@pytest.mark.xfail(
    reason="AMR is out of scope for the native V-cycle: GmgPrecondT is single-level and "
    "ginkgo_solve_composite takes no `precond` argument, so a composite (multi-level) "
    "system can only be solved UNPRECONDITIONED — its BiCGStab iteration count grows "
    "with the grid, which is exactly what a V-cycle would fix. Enabling it needs a "
    "`precond` argument on ginkgo_solve_composite plus a multi-level hierarchy in "
    "GmgPrecondT (restriction/prolongation across the coarse/fine interface).",
    strict=False,
)
def test_native_gmg_preconditions_composite_amr(blockamr_session):
    """A 2-level AMR system should be solvable with the native V-cycle as preconditioner.

    Covers real AMR on the way: the unpreconditioned composite solve below runs and is
    checked against MLMG's own composite solve on the same hierarchy (a coarse 8^3 grid
    with one centrally refined ratio-2 patch), so the multi-level operator, the
    coarse/fine interface and the covered-cell average_down are all exercised on every
    run. Only the last two assertions — that a GMG-preconditioned composite solve
    exists and needs fewer iterations — are what fails today.
    """
    plain, diffs = _run_composite_amr("cuda", solver="bicgstab")
    assert "unsupported" not in plain, plain.get("unsupported")
    assert plain["converged"] is True
    assert max(diffs) < 1e-6, f"unpreconditioned composite differs from MLMG: {diffs}"

    preconditioned, pc_diffs = _run_composite_amr("cuda", solver="bicgstab", precond="gmg")
    assert "unsupported" not in preconditioned, (
        f'precond="gmg" on a composite system: {preconditioned.get("unsupported")}'
    )
    assert max(pc_diffs) < 1e-6, f"preconditioned composite differs from MLMG: {pc_diffs}"
    assert preconditioned["num_iters"] < plain["num_iters"], (
        f"preconditioning did not help: {preconditioned['num_iters']} vs {plain['num_iters']}"
    )


# ---------------------------------------------------------------------------
# 11. Robin BCs — out of scope
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="out of scope: parseBc accepts periodic / dirichlet / neumann only, and the "
    "ghost fill is a sign reflection (bc.hpp bcGhostFill), which cannot express a "
    "Robin condition a*u + b*du/dn = g. MLMG has LinOpBCType::Robin. Adding it "
    "means a third ghost-fill branch plus an inhomogeneous term (the fold in "
    "_dirichlet_fold generalises), at which point this test becomes the "
    "manufactured-solution order check."
)
def test_robin_bc(blockamr_session):
    """A Robin boundary condition should reach second order on a manufactured solution."""
    pytest.fail("unreachable: bc=['robin', ...] is rejected by parseBc")
