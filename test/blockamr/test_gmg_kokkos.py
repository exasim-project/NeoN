# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The Kokkos GMG V-cycle must compute the same V-cycle as the AMReX one.

``bench/gmg_vcycle.cpp`` runs the native geometric-multigrid V-cycle of
``solvers/gmg_precond.hpp`` twice: once with its production AMReX kernels and once
with the Kokkos twins in ``bench/gmg_kokkos.hpp``. A timing comparison is only worth
reading if both really performed the V-cycle, and a multigrid V-cycle is unusually
easy to get *nearly* right -- a launcher that misses a colour, mismatches the
coarse-to-fine box mapping, or reads stale ghosts still returns a smaller residual,
just a worse one. So the gate is the residual reduction of ONE V-cycle from
``sol = 0``, which every one of those mistakes changes:

* it must actually drop (a V-cycle of this operator is a strong preconditioner);
* the two backends must agree on it to round-off, since they run the same arithmetic
  in the same order and differ only in the launcher.

Multi-box cases are the interesting ones twice over: they exercise the halo exchange
between colour sweeps, and because the hierarchy coarsens the BoxArray in place the
box count is preserved on every level -- so the coarse levels are many tiny boxes,
which is where a per-box launcher and a fused one diverge most.
"""

import numpy as np
import pytest

import blockamr

BACKENDS = ["amrex", "kokkos"]

# Small on purpose: this file proves the V-cycle is the same V-cycle, not that it is
# fast. benchmarks/blockamr/bench_gmg_kokkos.py carries the sizes that measure cost.
N_CELL = 16
MAX_SIZE = [None, 8]
MAX_SIZE_IDS = ["1box", "8box"]


def _mesh(max_size):
    box = blockamr.Box([0, 0, 0], [N_CELL - 1] * 3)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # triply periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(N_CELL if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_face(geom, dm, d, max_size, value):
    dom = geom.domain()
    fb = blockamr.Box(dom.small_end(), dom.big_end())
    fb.surrounding_nodes(d)
    fba = blockamr.BoxArray(fb)
    fba.max_size(max_size)
    mf = blockamr.MultiFab(fba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _problem(max_size):
    """Periodic Helmholtz (phi - laplacian phi) in face-coefficient form, plus a
    smooth mean-zero rhs. Returns everything the bench needs, alive for the call."""
    box_size = N_CELL if max_size is None else max_size
    geom, ba, dm = _mesh(max_size)
    dx = geom.cell_size()

    alpha = blockamr.MultiFab(ba, dm, 1, 0)
    alpha.set_val(1.0)
    faces = [_const_face(geom, dm, d, box_size, -1.0 / dx[d] ** 2) for d in range(3)]

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    rhs.set_val(0.0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        nx, ny, nz = arr.shape[:3]
        x = (lo[0] + np.arange(nx) + 0.5) * dx[0]
        y = (lo[1] + np.arange(ny) + 0.5) * dx[1]
        z = (lo[2] + np.arange(nz) + 0.5) * dx[2]
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
        arr[:, :, :, 0] = np.sin(2 * np.pi * xg) * np.sin(2 * np.pi * yg) * np.sin(2 * np.pi * zg)
        rhs.copy_from(mfi, arr)

    return geom, ba, dm, rhs, alpha, faces


def _vcycle(backend, max_size, **kwargs):
    geom, _, _, rhs, alpha, faces = _problem(max_size)
    return dict(
        blockamr.bench_gmg_vcycle(
            backend,
            geom,
            rhs,
            alpha,
            faces[0],
            faces[1],
            faces[2],
            iters=1,
            batches=1,
            **kwargs,
        )
    )


def test_backends_registered():
    assert blockamr.bench_gmg_backends() == BACKENDS


def test_unknown_backend_raises():
    with pytest.raises(RuntimeError, match="unknown backend"):
        _vcycle("nope", None)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_one_vcycle_reduces_the_residual(backend, max_size):
    r = _vcycle(backend, max_size)
    assert r["resid0"] > 0.0
    # A V-cycle on this operator is strong; anything close to 1.0 means the cycle
    # did not really run.
    assert r["resid1"] < 0.25 * r["resid0"]


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_kokkos_matches_amrex(max_size):
    """Same arithmetic in the same order, so the residuals must agree to round-off."""
    a = _vcycle("amrex", max_size)
    k = _vcycle("kokkos", max_size)
    assert k["resid0"] == pytest.approx(a["resid0"], rel=1e-12)
    assert k["resid1"] == pytest.approx(a["resid1"], rel=1e-10)


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_hierarchy_is_identical_and_preserves_box_count(max_size):
    """Both backends build the same hierarchy, and coarsening keeps the box count --
    the property that makes the coarse levels launch-bound."""
    a = _vcycle("amrex", max_size)
    k = _vcycle("kokkos", max_size)
    assert a["nlevels"] > 1
    assert k["nlevels"] == a["nlevels"]
    assert k["boxes_per_level"] == a["boxes_per_level"]
    assert k["cells_per_level"] == a["cells_per_level"]
    # One box count for every level, and cells falling 8x per level.
    assert len(set(a["boxes_per_level"])) == 1
    cells = a["cells_per_level"]
    assert all(cells[i + 1] == cells[i] // 8 for i in range(len(cells) - 1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_levels_truncates_the_hierarchy(backend):
    """max_levels is what a caller uses to stop before the tiny-box levels."""
    r = _vcycle(backend, 8, max_levels=2)
    assert r["nlevels"] == 2


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_bench_reproduces_the_production_vcycle(max_size):
    """The bench's amrex column must BE the production V-cycle, not a lookalike.

    ``solver="gmg"`` is the stationary iteration x <- x + V(b - A x), so with
    ``max_iter=1`` from x = 0 its reported residual over ||b|| is exactly one
    V-cycle's reduction -- the same number the bench reports as resid1/resid0. If
    these ever diverge, the bench is measuring a V-cycle that production does not
    run, and every timing in bench_gmg_kokkos.py is answering the wrong question.
    """
    solver_cls = getattr(blockamr, "FaceCoeffSolver", None)
    if solver_cls is None:
        pytest.skip("FaceCoeffSolver requires a Ginkgo build")

    bench = _vcycle("amrex", max_size, pre_sweeps=2, post_sweeps=2, coarsest_sweeps=8, omega=1.0)

    geom, ba, dm, rhs, alpha, faces = _problem(max_size)
    sol = blockamr.MultiFab(ba, dm, 1, 1)  # 1 ghost, as the bench's L0 sol
    sol.set_val(0.0)

    b = np.concatenate([rhs.copy_to_host(m)[:, :, :, 0].ravel() for m in blockamr.MFIterator(rhs)])
    solver = solver_cls(
        alpha=alpha,
        ux=faces[0],
        lx=faces[0],
        uy=faces[1],
        ly=faces[1],
        uz=faces[2],
        lz=faces[2],
        geom=geom,
        executor="cuda",
        solver="gmg",
        max_iter=1,
        rtol=0.0,
        gmg_pre_sweeps=2,
        gmg_post_sweeps=2,
        gmg_coarsest_sweeps=8,
        gmg_omega=1.0,
        precond_cycles=1,
    )
    st = solver.solve(rhs, sol)
    production = st["res_norm"] / float(np.linalg.norm(b))

    assert bench["resid1"] / bench["resid0"] == pytest.approx(production, rel=1e-10)


@pytest.mark.parametrize("max_size", MAX_SIZE, ids=MAX_SIZE_IDS)
def test_single_level_still_smooths(max_size):
    """max_levels=1 is the smoother alone: no restriction, no prolongation. It must
    still reduce the residual, and both backends must agree -- this isolates the
    colour sweep from the inter-level kernels."""
    a = _vcycle("amrex", max_size, max_levels=1)
    k = _vcycle("kokkos", max_size, max_levels=1)
    assert a["nlevels"] == 1
    assert a["resid1"] < a["resid0"]
    assert k["resid1"] == pytest.approx(a["resid1"], rel=1e-10)
