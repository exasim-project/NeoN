# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Native geometric-multigrid preconditioner (precond="gmg") for FaceCoeffSolver.

Unlike ``precond_mlmg`` (gap 1a, which wraps an AMReX MLMG), ``precond="gmg"``
is a V-cycle built ONCE from AMReX primitives directly on the face-coefficient
operator — no MLLinOp/MLMG anywhere in its path: coarsen-by-2 levels with
rediscretised coefficients, symmetric red-black Gauss-Seidel smoothing,
volume-average restriction and piecewise-constant prolongation. Model problem:
periodic Helmholtz
(phi - laplacian phi) — diagonal source alpha=1, symmetric face coefficients
-1/dx^2 — with a seeded random rhs so CG must work across the whole spectrum.
"""

import numpy as np
import pytest

import blockamr


def _make_mesh(n, periodic=True):
    """Single-box mesh on [0,1]^3 with n cells per side."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    per = [1, 1, 1] if periodic else [0, 0, 0]
    geom = blockamr.Geometry(box, rb, 0, per)
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
    """alpha=1 cell source + symmetric -1/dx^2 face coeffs (Helmholtz)."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 1.0)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _make_solver_or_skip(coeffs, geom, executor, cls="FaceCoeffSolver", **kwargs):
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
def test_gmg_matches_unpreconditioned(blockamr_session, executor):
    """gmg-preconditioned CG solves the SAME matrix: solutions agree to < 1e-6.

    Helmholtz is nonsingular, so the gmg solve and the unpreconditioned solve
    of the identical face-coefficient operator agree directly.
    """
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    s_gmg = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=100, rtol=1e-11, precond="gmg"
    )
    sol_gmg = _zero_sol(ba, dm)
    stats = s_gmg.solve(rhs, sol_gmg)
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6

    s_plain = _make_solver_or_skip(coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-11)
    sol_plain = _zero_sol(ba, dm)
    stats_plain = s_plain.solve(rhs, sol_plain)
    assert stats_plain["converged"] is True

    max_diff = _max_abs_diff(sol_gmg, sol_plain)
    assert max_diff < 1e-6, f"Max |sol_gmg - sol_plain| = {max_diff} exceeds 1e-6"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_iterations_flat_in_n(blockamr_session, executor):
    """The gmg-preconditioned iteration count is small and flat in N (the MG win).

    Unpreconditioned CG grows ~O(1/h) (observed 128 -> 250 for N=32 -> 64);
    the native V-cycle drops it to 9 at BOTH sizes (observed).
    """
    iters = {}
    for n in (32, 64):
        geom, ba, dm = _make_mesh(n)
        coeffs = _helmholtz_coeffs(geom, ba, dm, n)
        rhs = _random_rhs(ba, dm)

        s_gmg = _make_solver_or_skip(
            coeffs, geom, executor, solver="cg", max_iter=100, rtol=1e-10, precond="gmg"
        )
        stats = s_gmg.solve(rhs, _zero_sol(ba, dm))
        assert stats["converged"] is True
        iters[n] = stats["num_iters"]

        s_plain = _make_solver_or_skip(
            coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-10
        )
        stats_plain = s_plain.solve(rhs, _zero_sol(ba, dm))
        assert stats_plain["converged"] is True
        # (i) far below unpreconditioned at the same N (observed 9 vs 128+).
        assert 5 * stats["num_iters"] < stats_plain["num_iters"], (
            f"N={n}: gmg {stats['num_iters']} not << unpreconditioned {stats_plain['num_iters']}"
        )

    # (ii) flat in N (observed: 9 at both sizes).
    assert iters[64] <= iters[32] + 5, f"iteration count grows with N: {iters}"
    assert iters[32] <= 25 and iters[64] <= 25, f"iteration count not small: {iters}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_cycles_reduce_iters(blockamr_session, executor):
    """More V-cycles per application make each CG iteration stronger (fewer iters)."""
    N = 32
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    counts = {}
    for cycles in (1, 2):
        s = _make_solver_or_skip(
            coeffs,
            geom,
            executor,
            solver="cg",
            max_iter=100,
            rtol=1e-10,
            precond="gmg",
            precond_cycles=cycles,
        )
        stats = s.solve(rhs, _zero_sol(ba, dm))
        assert stats["converged"] is True
        counts[cycles] = stats["num_iters"]
    assert counts[2] < counts[1], f"2 cycles ({counts[2]}) not fewer iters than 1 ({counts[1]})"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_dirichlet(blockamr_session, executor):
    """gmg composes with non-periodic BCs: all-Dirichlet Poisson, N=32.

    The BC ghost reflection coarsens cleanly, so the same bc spec applies on
    every level: the gmg solve matches its own unpreconditioned solve to
    < 1e-6 and converges in a flat/low count.
    """
    N = 32
    geom, ba, dm = _make_mesh(N, periodic=False)
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 0.0)  # pure Poisson, Dirichlet -> nonsingular
    fx = _const_face(geom, dm, 0, N, -inv_dx2)
    fy = _const_face(geom, dm, 1, N, -inv_dx2)
    fz = _const_face(geom, dm, 2, N, -inv_dx2)
    coeffs = (alpha, fx, fy, fz)
    bc = ["dirichlet"] * 6
    rhs = _random_rhs(ba, dm)

    s_gmg = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=100, rtol=1e-11, bc=bc, precond="gmg"
    )
    sol_gmg = _zero_sol(ba, dm)
    stats = s_gmg.solve(rhs, sol_gmg)
    assert stats["converged"] is True
    assert stats["num_iters"] <= 30, f"Dirichlet gmg count not low: {stats['num_iters']}"

    s_plain = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-11, bc=bc
    )
    sol_plain = _zero_sol(ba, dm)
    stats_plain = s_plain.solve(rhs, sol_plain)
    assert stats_plain["converged"] is True
    assert stats["num_iters"] < stats_plain["num_iters"]

    max_diff = _max_abs_diff(sol_gmg, sol_plain)
    assert max_diff < 1e-6, f"Max |sol_gmg - sol_plain| = {max_diff} exceeds 1e-6"


def test_gmg_validation_errors(blockamr_session):
    """precond='gmg' on the CSR solver raises; gmg + precond_mlmg raises."""
    N = 8
    geom, ba, dm = _make_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)

    with pytest.raises(RuntimeError, match="matrix-free only"):
        _make_solver_or_skip(
            coeffs, geom, "reference", cls="FaceCoeffCsrSolver", solver="cg", precond="gmg"
        )

    mlmg_holder = []  # an MLMG needs an MLABecLaplacian; build the minimal one
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3,
        [blockamr.LinOpBCType.Periodic] * 3,
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
    mlmg_holder.append(blockamr.MLMG(abec))
    with pytest.raises(RuntimeError, match="cannot be combined"):
        _make_solver_or_skip(
            coeffs,
            geom,
            "reference",
            solver="cg",
            precond="gmg",
            precond_mlmg=mlmg_holder[0],
        )

    with pytest.raises(RuntimeError, match="unknown precond"):
        _make_solver_or_skip(coeffs, geom, "reference", solver="cg", precond="bogus")
