# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""2-level composite AMR solve through the matrix-free Ginkgo path.

``ginkgo_solve_composite`` solves the multi-level COMPOSITE MLLinOp system:
the Ginkgo vector concatenates all levels' cells and the mat-vec is the
multi-level ``MLMG::apply`` (coarse/fine interface interpolation, reflux and
covered-cell average_down all handled by AMReX), so the solved system is
identical to MLMG's own composite solve. Coarse cells covered by the fine
patch are slaved, not DOFs — their operator columns are zero and their final
values are the average_down of the fine solution (MLMG's convention).

The composite operator is NOT symmetric (the coarse/fine ghost interpolation
is not the adjoint of the reflux), so the solver is BiCGStab; plain CG was
measured to diverge on this hierarchy (residual 2e3 after 4000 iterations).

Model problem: periodic Helmholtz (alpha=1, beta=1 MLABecLaplacian — positive
definite, no nullspace) on a coarse 32^3 grid with one centrally-refined
ratio-2 patch (coarse cells [8,23]^3 -> fine 32^3 box), seeded random rhs so
the Krylov solver works across the whole spectrum.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

N = 32  # coarse cells per side
PATCH_LO, PATCH_HI = 8, 23  # coarse index range of the refined patch (ratio 2)


def _make_hierarchy():
    """(geom, ba, dm, cell_box) per level for the 2-level periodic hierarchy."""
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    box_c = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    geom_c = blockamr.Geometry(box_c, rb, 0, [1, 1, 1])
    ba_c = blockamr.BoxArray(box_c)
    ba_c.max_size(N)
    dm_c = blockamr.DistributionMapping(ba_c)

    box_f_dom = blockamr.Box([0, 0, 0], [2 * N - 1, 2 * N - 1, 2 * N - 1])
    geom_f = blockamr.Geometry(box_f_dom, rb, 0, [1, 1, 1])
    patch = blockamr.Box([2 * PATCH_LO] * 3, [2 * PATCH_HI + 1] * 3)
    ba_f = blockamr.BoxArray(patch)
    ba_f.max_size(2 * N)
    dm_f = blockamr.DistributionMapping(ba_f)
    return (geom_c, ba_c, dm_c, box_c), (geom_f, ba_f, dm_f, patch)


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(cell_box, dm, d, value):
    face_box = blockamr.Box(cell_box.small_end(), cell_box.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(2 * N + 1)  # single box -> matches the cell dm
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _make_abec(levels):
    """Composite MLABecLaplacian: alpha=1, beta=1, unit coefficients, periodic."""
    abec = blockamr.MLABecLaplacian(
        [lv[0] for lv in levels], [lv[1] for lv in levels], [lv[2] for lv in levels]
    )
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3,
        [blockamr.LinOpBCType.Periodic] * 3,
    )
    abec.set_scalars(1.0, 1.0)
    for lev, (_geom, ba, dm, cell_box) in enumerate(levels):
        abec.set_level_bc(lev, None)
        abec.set_a_coeffs(lev, _const_cell(ba, dm, 1.0))
        abec.set_b_coeffs(
            lev,
            _const_face(cell_box, dm, 0, 1.0),
            _const_face(cell_box, dm, 1, 1.0),
            _const_face(cell_box, dm, 2, 1.0),
        )
    return abec


def _random_rhs(ba, dm, seed):
    """Cell MultiFab with seeded random values — full spectrum, so Krylov must work."""
    rng = np.random.default_rng(seed)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        rhs.copy_from(mfi, arr)
    return rhs


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _max_abs_diff(a, b):
    """Max-norm difference between the valid regions of two cell MultiFabs."""
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


def _composite_solve_or_skip(lp, sol, rhs, executor, **kwargs):
    """Call ginkgo_solve_composite, skipping if Ginkgo/CUDA are unavailable."""
    if not hasattr(blockamr, "ginkgo_solve_composite"):
        pytest.skip("blockamr.ginkgo_solve_composite binding not available")
    try:
        return blockamr.ginkgo_solve_composite(lp, sol, rhs, executor=gko_executor(executor), **kwargs)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise


def _mlmg_reference_solve(levels, rhs_c, rhs_f):
    """Tight-tolerance MLMG composite solve on the hierarchy -> (sol_c, sol_f)."""
    abec = _make_abec(levels)
    sol_c = _zero_sol(levels[0][1], levels[0][2])
    sol_f = _zero_sol(levels[1][1], levels[1][2])
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve([sol_c, sol_f], [rhs_c, rhs_f], 1e-12, 0.0)
    return sol_c, sol_f


def _average_down(levels, sol_f, sol_c):
    """Coarse covered cells <- average of the fine solution (both conventions)."""
    blockamr.average_down(
        sol_f, sol_c, levels[1][0], levels[0][0], 0, 1, blockamr.IntVect(2, 2, 2)
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_composite_matches_mlmg(blockamr_session, executor):
    """Ginkgo composite BiCGStab matches MLMG's composite solve to < 1e-6.

    Fine level compared directly; coarse level compared after average_down of
    BOTH solutions (covered coarse cells are slaved to the fine level, so only
    their averaged-down representative is well-defined).
    """
    levels = _make_hierarchy()
    (_geom_c, ba_c, dm_c, _box_c), (_geom_f, ba_f, dm_f, _patch) = levels
    rhs_c = _random_rhs(ba_c, dm_c, seed=11)
    rhs_f = _random_rhs(ba_f, dm_f, seed=12)

    sol_ref_c, sol_ref_f = _mlmg_reference_solve(levels, rhs_c, rhs_f)

    abec_gko = _make_abec(levels)
    sol_gko_c = _zero_sol(ba_c, dm_c)
    sol_gko_f = _zero_sol(ba_f, dm_f)
    stats = _composite_solve_or_skip(
        abec_gko,
        [sol_gko_c, sol_gko_f],
        [rhs_c, rhs_f],
        executor,
        max_iter=2000,
        rtol=1e-12,
        sign=+1.0,
        solver="bicgstab",
    )

    # Stats sanity (case c).
    assert stats["num_iters"] > 0
    assert stats["converged"] is True
    assert stats["res_norm"] < 1e-6, f"Residual norm {stats['res_norm']} too large"
    assert len(stats["res_history"]) > 0

    # Case b: per-level agreement. Coarse after average_down of both.
    max_diff_fine = _max_abs_diff(sol_gko_f, sol_ref_f)
    assert max_diff_fine < 1e-6, f"Max fine |gko - mlmg| = {max_diff_fine} exceeds 1e-6"

    _average_down(levels, sol_ref_f, sol_ref_c)
    _average_down(levels, sol_gko_f, sol_gko_c)
    max_diff_crse = _max_abs_diff(sol_gko_c, sol_ref_c)
    assert max_diff_crse < 1e-6, f"Max coarse |gko - mlmg| = {max_diff_crse} exceeds 1e-6"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_composite_warm_start(blockamr_session, executor):
    """A second composite solve on an already-converged sol stops immediately.

    Proves the incoming per-level values seed the initial guess (residual-
    correction form, as in ginkgo_solve).
    """
    levels = _make_hierarchy()
    (_geom_c, ba_c, dm_c, _box_c), (_geom_f, ba_f, dm_f, _patch) = levels
    rhs_c = _random_rhs(ba_c, dm_c, seed=21)
    rhs_f = _random_rhs(ba_f, dm_f, seed=22)

    abec = _make_abec(levels)
    sol_c = _zero_sol(ba_c, dm_c)
    sol_f = _zero_sol(ba_f, dm_f)
    stats_cold = _composite_solve_or_skip(
        abec,
        [sol_c, sol_f],
        [rhs_c, rhs_f],
        executor,
        max_iter=2000,
        rtol=1e-10,
        sign=+1.0,
    )
    assert stats_cold["converged"] is True
    assert stats_cold["num_iters"] > 5, "Cold start converged suspiciously fast"

    stats_warm = _composite_solve_or_skip(
        abec,
        [sol_c, sol_f],
        [rhs_c, rhs_f],
        executor,
        max_iter=2000,
        rtol=1e-10,
        sign=+1.0,
    )
    assert stats_warm["num_iters"] <= 5, (
        f"Warm start took {stats_warm['num_iters']} iterations — initial guess ignored?"
    )
