# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""MLMG-preconditioned matrix-free Ginkgo Krylov solve (precond_mlmg).

The persistent ``FaceCoeffSolver`` accepts a caller-supplied ``MLMG`` built on
an equivalent operator; each Krylov iteration is then preconditioned by a fixed
small number of multigrid V-cycles (``precond_cycles``). This is the decisive
scaling fix: unpreconditioned CG iterations grow ~O(1/h) with resolution while
the preconditioned count stays flat (multigrid), so the matrix-free path wins
at scale. Model problem: periodic Helmholtz (phi - laplacian phi) — diagonal
source alpha=1, symmetric face coefficients -1/dx^2 — with a seeded random rhs
so the Krylov solver must work across the whole spectrum.
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


def _helmholtz_coeffs(geom, ba, dm, n, a=1.0):
    """alpha=a cell source + symmetric -1/dx^2 face coeffs (periodic Helmholtz)."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, a)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _make_abec(geom, ba, dm, n, a_scalar=1.0):
    """MLABecLaplacian equivalent of the face-coefficient Helmholtz operator."""
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3,
        [blockamr.LinOpBCType.Periodic] * 3,
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(a_scalar, 1.0)
    abec.set_a_coeffs(0, _const_cell(ba, dm, 1.0))
    abec.set_b_coeffs(
        0,
        _const_face(geom, dm, 0, n, 1.0),
        _const_face(geom, dm, 1, n, 1.0),
        _const_face(geom, dm, 2, n, 1.0),
    )
    return abec


def _make_solver_or_skip(coeffs, geom, executor, **kwargs):
    """Construct a FaceCoeffSolver, skipping if Ginkgo/CUDA are unavailable."""
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
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _precond_solve(geom, ba, dm, n, coeffs, rhs, executor, **kwargs):
    """Preconditioned CG solve; returns (stats, sol). MLMG stays alive via keep_alive."""
    mlmg = blockamr.MLMG(_make_abec(geom, ba, dm, n, a_scalar=kwargs.pop("a_scalar", 1.0)))
    s = _make_solver_or_skip(
        coeffs, geom, executor, solver="cg", precond_mlmg=mlmg, **kwargs
    )
    sol = _zero_sol(ba, dm)
    stats = s.solve(rhs, sol)
    return stats, sol


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_precond_matches_mlmg(blockamr_session, executor):
    """Preconditioned matrix-free CG matches a plain MLMG solve of the same matrix.

    Helmholtz is nonsingular, so the solutions agree directly (no mean-free
    comparison needed). The residual is monotone — the (approximately
    symmetric) V-cycle does not upset CG here.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    stats, sol_pc = _precond_solve(
        geom, ba, dm, N, coeffs, rhs, executor, max_iter=100, rtol=1e-11
    )
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6

    hist = stats["res_history"]
    assert all(b <= a for a, b in zip(hist, hist[1:])), f"non-monotone residual: {hist}"

    # Reference: the identical operator solved by MLMG directly.
    sol_ref = _zero_sol(ba, dm)
    mlmg_ref = blockamr.MLMG(_make_abec(geom, ba, dm, N))
    mlmg_ref.set_verbose(0)
    mlmg_ref.set_max_iter(200)
    mlmg_ref.solve(sol_ref, rhs, 1e-11, 1e-13)

    max_diff = _max_abs_diff(sol_pc, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_precond - sol_mlmg| = {max_diff} exceeds 1e-6"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_precond_iterations_flat_in_n(blockamr_session, executor):
    """The preconditioned iteration count is small and ~flat in N (the MG win).

    Unpreconditioned CG grows ~O(1/h) (observed 128 -> 250 iters for N=32 ->
    64); one V-cycle of preconditioning drops it to ~7 at BOTH sizes.
    """
    iters = {}
    for n in (32, 64):
        geom, ba, dm = _make_periodic_mesh(n)
        coeffs = _helmholtz_coeffs(geom, ba, dm, n)
        rhs = _random_rhs(ba, dm)

        stats, _ = _precond_solve(
            geom, ba, dm, n, coeffs, rhs, executor, max_iter=100, rtol=1e-10
        )
        assert stats["converged"] is True
        iters[n] = stats["num_iters"]

        s_plain = _make_solver_or_skip(
            coeffs, geom, executor, solver="cg", max_iter=2000, rtol=1e-10
        )
        stats_plain = s_plain.solve(rhs, _zero_sol(ba, dm))
        assert stats_plain["converged"] is True
        # (i) much lower than unpreconditioned at the same N (observed 7 vs 128+).
        assert 5 * stats["num_iters"] < stats_plain["num_iters"], (
            f"N={n}: preconditioned {stats['num_iters']} not << "
            f"unpreconditioned {stats_plain['num_iters']}"
        )

    # (ii) roughly flat in N (observed: 7 at both sizes).
    assert iters[64] <= iters[32] + 3, f"iteration count grows with N: {iters}"
    assert iters[32] <= 20 and iters[64] <= 20, f"iteration count not small: {iters}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_precond_cycles_reduce_iters(blockamr_session, executor):
    """More V-cycles per application make each CG iteration stronger (fewer iters)."""
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N)
    rhs = _random_rhs(ba, dm)

    stats1, _ = _precond_solve(
        geom, ba, dm, N, coeffs, rhs, executor, max_iter=100, rtol=1e-10, precond_cycles=1
    )
    stats2, _ = _precond_solve(
        geom, ba, dm, N, coeffs, rhs, executor, max_iter=100, rtol=1e-10, precond_cycles=2
    )
    assert stats1["converged"] is True and stats2["converged"] is True
    assert stats2["num_iters"] < stats1["num_iters"], (
        f"2 cycles ({stats2['num_iters']}) not fewer iters than 1 ({stats1['num_iters']})"
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_precond_with_project_nullspace(blockamr_session, executor):
    """Smoke: the preconditioner coexists with project_nullspace (singular Poisson).

    alpha=0 makes the fully-periodic operator singular (constant nullspace);
    the preconditioner MLMG is built on the matching a_scalar=0 MLABecLaplacian
    (AMReX handles the singularity internally). The solve stays convergent and
    returns the mean-zero representative.
    """
    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    coeffs = _helmholtz_coeffs(geom, ba, dm, N, a=0.0)
    rhs = _random_rhs(ba, dm)
    # Consistent rhs: subtract the mean exactly.
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] -= float(np.mean(arr[:, :, :, 0]))
        rhs.copy_from(mfi, arr)

    stats, sol = _precond_solve(
        geom, ba, dm, N, coeffs, rhs, executor,
        max_iter=200, rtol=1e-9, project_nullspace=True, a_scalar=0.0,
    )
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-5

    means = [float(np.mean(sol.copy_to_host(mfi)[:, :, :, 0])) for mfi in blockamr.MFIterator(sol)]
    assert abs(np.mean(means)) < 1e-10, f"solution mean {np.mean(means)} not ~0"
