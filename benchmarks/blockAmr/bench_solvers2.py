# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Solver comparison through the type-erased ``blockamr.linear_algebra`` seam.

The counterpart of ``bench_solvers.py``, which drives the LEGACY
``FaceCoeffSolver`` / ``FaceCoeffCsrSolver`` facades. Here the operator is
assembled the way a discretiser would assemble it —
``MFFaceCoeffs.symmetric(MeshLevel(...))`` + ``Matrix.diagonal_source(alpha)`` +
``system += laplacian(gamma, geom, bc=...)`` — and handed to
``la.Solver(SolverConfig(...))``, which never learns what format it got.

The same symmetric Helmholtz operator (``alpha*phi - div(gamma grad phi)``,
alpha = gamma = 1) is solved by every row, so the table is a like-for-like
comparison of Krylov methods and preconditioners *at a fixed operator*.

Rows THROUGH the seam (``seam=yes``)::

    la-cg / la-bicgstab / la-gmres / la-gcr / la-fcg   solver=..., precond="none"
    la-cg-gmg                                          precond="gmg"
    la-cg-gmgk                                         precond="gmg_kokkos"
    la-cg-mlmg                                         precond="mlmg"

Rows NOT through the seam (``seam=no``), and the table says so per row because
the distinction IS part of the result::

    mlmg        AMReX geometric multigrid built DIRECTLY on an MLABecLaplacian,
                exactly as bench_solvers.py's `mlmg` method builds it. It is the
                performance REFERENCE — the speedup column is t_mlmg/t_method —
                and it is not reachable from behind the seam today (see below).
    legacy-mf   the same operator through the legacy `FaceCoeffSolver` CG on
                hand-built coefficients (alpha = 1, face = -gamma/dx**2, the BCs
                declared with `bc=` and folded by the solver). The ANCHOR for
                "does the operator seam cost anything against writing the
                coefficients by hand?" — same Krylov method, same matrix-free
                mat-vec, only the assembly route differs. Building it is part of
                `setup_ms`, outside the timed region.

Deliberately ABSENT: the assembled CSR format. ``bench_solvers.py``'s ``csr`` row
is the matrix-free-vs-assembled comparison and stays there.

Also absent, because the seam refuses them by design: ``solver="gmg"/"ir"/"mpir"``
want the hierarchy as the SOLVER rather than as a preconditioner of one. Use
``FaceCoeffSolver`` (``bench_solvers.py``'s ``gmg`` / ``gmg-ir`` / ``mf-mpir``).

WHY THE DIRICHLET CONFIGURATION EXISTS
--------------------------------------
``bench_solvers.py`` benchmarks a triply-PERIODIC problem. On a periodic domain
``ops::Laplacian`` folds nothing, so the whole boundary-condition question is
invisible there. ``blockAmr`` stores a non-periodic BC FOLDED: the domain-face
coefficient is zeroed and its contribution summed into the cell-centred diagonal
source. ``plans/blockamr-mlmg-fold-coarsening.md`` measured what that costs AMReX
MLMG when the folded form is what MLMG is handed: **2.1x-2.6x more V-cycles**
(8 -> 17, 8 -> 20, 9 -> 23 at 32/64/128 cubed, Dirichlet), because
``average_down`` treats the folded ``2*gamma/dx**2`` term as a dx-independent
physical ``a`` and so leaves every coarse level's boundary diagonal 2x too strong.

This benchmark is meant to become the ACCEPTANCE GATE for the future "unfold"
slice, whose criterion is that MLMG's V-cycle count drops back to the
physical-form value while every matrix-free row is unchanged. That is only
measurable on a Dirichlet domain, hence ``--bc {periodic,dirichlet,both}``
(default: both) and hence ``iters`` being a first-class column on every row at
every size — so the future comparison is a diff of this table rather than a new
experiment.

WHAT THE NUMBERS CANNOT SAY
---------------------------
* ``iters`` is each solver's NATURAL unit and the units differ: ``mlmg`` reports
  V-cycles (few, ~flat in N), every other row reports fine-grid Krylov
  iterations. They are comparable within a column, not across the ``mlmg`` row.
* The convergence NORM differs too. MLMG stops on ``||r||_inf <= rtol*||b||_inf``
  and is not configurable; the Ginkgo-side rows stop on the 2-norm by default.
  ``--norm linf`` puts them on MLMG's criterion, which is the apples-to-apples
  setting for the ``iters`` column; ``--norm l2`` (the default) keeps each side's
  native convention and makes the ``mlmg`` iteration count only indicative.
* ``la.Solver`` is STATELESS: ``solve()`` asks the matrix for a preconditioner and
  constructs a fresh Ginkgo solver on EVERY call (``la/solver.hpp``). So the
  ``la-cg-gmg`` / ``la-cg-gmgk`` rows re-pay the whole GMG hierarchy build inside
  every timed solve, where ``bench_solvers.py``'s ``mf-gmg`` / ``mf-gmgk`` build it
  once in the constructor and charge it to ``setup_ms``. Their wall clocks are
  therefore NOT comparable across the two benchmarks; their iteration counts are.
  ``setup_ms`` here is only what a caller does before the first solve.
* Wall clock is a shared-GPU quantity. Iteration counts are deterministic and
  survive contention; ``solve_ms`` does not.
* Single box, single GPU, single AMR level, constant coefficients, isotropic
  cells. Nothing here says anything about multi-box, MPI or composite hierarchies.

Fairness: identical operator, identical zero initial guess, identical tolerance
and identical right-hand side on every row. The rhs is a fixed-seed random draw
rather than ``bench_solvers.py``'s sum of sine modes, and that divergence is
deliberate — see :func:`seeded_rhs`, where the measurement that forced it is
recorded.

The V-cycle shape is ``GmgConfig``'s measured default (2+2 RB-GS sweeps, 16
coarsest sweeps, omega = 1.1) for every GMG row; it is deliberately not exposed as
a flag, so the table compares preconditioners rather than tunings.

Run::

    python benchmarks/blockAmr/bench_solvers2.py --n-cell 32 64 --bc both
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import numpy as np

import blockamr
import neon
from blockamr.linear_algebra import LinearSystem, MFFaceCoeffs, Solver, SolverConfig, laplacian

# (solver, precond) for every row that goes THROUGH blockamr.linear_algebra.
LA_METHODS = {
    "la-cg": ("cg", "none"),
    "la-bicgstab": ("bicgstab", "none"),
    "la-gmres": ("gmres", "none"),
    "la-gcr": ("gcr", "none"),
    "la-fcg": ("fcg", "none"),
    "la-cg-gmg": ("cg", "gmg"),
    "la-cg-gmgk": ("cg", "gmg_kokkos"),
    "la-cg-mlmg": ("cg", "mlmg"),
}
# Reference first (the speedup baseline), then the hand-built anchor, then the
# seam. `mlmg` and `legacy-mf` are NOT behind the interface and the `seam` column
# says so on every line -- presenting them uniformly would imply the seam reaches
# MLMG, which it does not and cannot until the unfold slice.
METHODS = ("mlmg", "legacy-mf", *LA_METHODS)

# Convergence norm for the Ginkgo-side rows; "l2" is their native convention,
# "linf" is MLMG's own criterion. The `mlmg` row ALWAYS stops on linf -- that is
# not configurable in MLMG -- so --norm linf is what makes the `iters` column
# comparable against it. Set from --norm.
NORM = "l2"


# ---------------------------------------------------------------------------
# Mesh + field helpers, copied from bench_solvers.py (single box -> face fabs
# align 1:1 with the cell fab; same decomposition for every method). Copied
# rather than imported: two benchmark scripts that import each other cannot be
# run or edited independently, and `build_mesh` needed the periodicity argument.
# ---------------------------------------------------------------------------
def build_mesh(n_cell: int, max_size: int, periodic: bool):
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1] if periodic else [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def const_face(geom, dm, d, max_size, value):
    dom = geom.domain()
    fb = blockamr.Box(dom.small_end(), dom.big_end())
    fb.surrounding_nodes(d)
    fba = blockamr.BoxArray(fb)
    fba.max_size(max_size)
    mf = blockamr.MultiFab(fba, dm, 1, 0)
    mf.set_val(value)
    return mf


def solution_to_host(mf):
    return np.concatenate(
        [mf.copy_to_host(mfi)[:, :, :, 0].ravel() for mfi in blockamr.MFIterator(mf)]
    )


# ---------------------------------------------------------------------------
# The three ways of building the SAME operator.
# ---------------------------------------------------------------------------
def make_abec(geom, ba, dm, max_size, periodic):
    """Helmholtz (phi - lap phi) as an MLABecLaplacian: alpha=beta=a=b=1.

    The Dirichlet branch calls `set_max_order(2)`, which is load-bearing rather
    than cosmetic. AMReX defaults to 3 (quadratic boundary interpolation), a
    genuinely more accurate and therefore DIFFERENT matrix from the two-point
    `ghost = -interior` that `ops::Laplacian`'s fold encodes. Without the detune
    this row would solve a different system and the parity check would fail —
    which is exactly what `plans/blockamr-mlmg-fold-coarsening.md` §2 established
    and what `test_ginkgo_bc.py::test_dirichlet_matches_mlmg` already relies on.
    """
    kind = blockamr.LinOpBCType.Periodic if periodic else blockamr.LinOpBCType.Dirichlet
    abec = blockamr.MLABecLaplacian(geom, ba, dm, blockamr.LPInfo())
    abec.set_domain_bc([kind] * 3, [kind] * 3)
    if not periodic:
        abec.set_max_order(2)
    abec.set_level_bc(0, None)  # homogeneous
    abec.set_scalars(1.0, 1.0)
    abec.set_a_coeffs(0, const_cell(ba, dm, 1.0))
    abec.set_b_coeffs(
        0,
        const_face(geom, dm, 0, max_size, 1.0),
        const_face(geom, dm, 1, max_size, 1.0),
        const_face(geom, dm, 2, max_size, 1.0),
    )
    return abec


def make_la_system(geom, ba, dm, bc, rhs, executor):
    """Assemble the Helmholtz system through the seam; returns (system, keepalive).

    Everything the system reads is returned with it. `LinearSystem` is NON-OWNING
    (the matrix and the rhs must outlive it) and `laplacian()` holds `gamma` BY
    POINTER, read at `+=` time — so `keepalive` is not decoration, it is what
    stops a use-after-free.
    """
    gamma = const_cell(ba, dm, 1.0)
    alpha = const_cell(ba, dm, 1.0)
    matrix = MFFaceCoeffs.symmetric(blockamr.MeshLevel(ba, dm, geom), executor=executor, bc=bc)
    matrix.diagonal_source(alpha)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma, geom, bc=bc)
    return system, (matrix, gamma, alpha)


def make_legacy_solver(geom, ba, dm, max_size, bc, rtol, atol, max_iter):
    """The legacy facade on hand-written coefficients: alpha=1, face=-gamma/dx**2.

    The BCs are DECLARED (`bc=`) and folded by the solver, which is the legacy
    equivalent of what `laplacian(bc=...)` folds into the coefficients — so this
    and the `la-cg` row are the same matrix, reached two different ways.
    """
    dx = geom.cell_size()
    alpha = const_cell(ba, dm, 1.0)
    fx = const_face(geom, dm, 0, max_size, -1.0 / dx[0] ** 2)
    fy = const_face(geom, dm, 1, max_size, -1.0 / dx[1] ** 2)
    fz = const_face(geom, dm, 2, max_size, -1.0 / dx[2] ** 2)
    return blockamr.FaceCoeffSolver(
        alpha=alpha,
        ux=fx,
        lx=fx,
        uy=fy,
        ly=fy,
        uz=fz,
        lz=fz,
        geom=geom,
        executor=neon.GPUExecutor(),
        solver="cg",
        bc=bc,
        max_iter=max_iter,
        rtol=rtol,
        atol=atol,
        norm=NORM,
    )


# ---------------------------------------------------------------------------
# One row: build once (setup), then time `repeats` solves from a zero guess.
# Every returned residual is a host scalar, which forces device completion, so
# the enclosing perf_counter span is a full solve.
# ---------------------------------------------------------------------------
def make_solve(method, geom, ba, dm, max_size, periodic, bc, rhs, sol, rtol, atol, max_iter):
    """Build whatever `method` needs; return (solve, keepalive).

    `keepalive` holds every AMReX/Ginkgo object the returned closure reads. It
    must be released by the caller before `blockamr.runtime()` exits: a
    destructor running after amrex::Finalize aborts with CUDA error 709.
    """
    if method == "mlmg":
        # Rebuilt per solve, exactly as bench_solvers.py's `mlmg` does, so this
        # row's setup_ms is its first solve and every repeat re-pays the build.
        def solve():
            sol.set_val(0.0)
            mlmg = blockamr.MLMG(make_abec(geom, ba, dm, max_size, periodic))
            mlmg.set_verbose(0)
            mlmg.set_max_iter(max_iter)
            mlmg.set_bottom_solver("cg")  # SPD
            mlmg.set_bottom_max_iter(max_iter)
            res = mlmg.solve(sol, rhs, rtol, atol)
            return mlmg.get_num_iters(), res  # V-cycles, not Krylov iterations

        return solve, ()

    if method == "legacy-mf":
        obj = make_legacy_solver(geom, ba, dm, max_size, bc, rtol, atol, max_iter)

        def solve():
            sol.set_val(0.0)
            st = obj.solve(rhs, sol)
            return st["num_iters"], st["res_norm"]

        return solve, (obj,)

    solver_kind, precond = LA_METHODS[method]
    system, keepalive = make_la_system(geom, ba, dm, bc, rhs, neon.GPUExecutor())
    la_solver = Solver(
        SolverConfig(
            solver=solver_kind,
            precond=precond,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
            norm=NORM,
        )
    )

    def solve():
        sol.set_val(0.0)
        st = la_solver.solve(system, sol)
        return st["num_iters"], st["res_norm"]

    return solve, (system, la_solver, *keepalive)


@dataclass
class Result:
    method: str
    bc: str
    n_cell: int
    seam: bool
    iters: int
    setup_ms: float
    solve_ms: float
    mcell_per_s: float
    final_res: float
    status: str
    snapshot: np.ndarray | None


def seeded_rhs(ba, dm, seed=7):
    """A fixed-seed random right-hand side — the ONE place this diverges from bench_solvers.py.

    ``bench_solvers.py`` uses a sum of a few sine/cosine modes. Every one of them
    is an exact eigenfunction of the PERIODIC Helmholtz operator, so the rhs lies
    in a four-dimensional eigenspace and CG lands on the answer in 6 iterations at
    every mesh size — measured, on the first run of this script. A row that takes
    6 iterations cannot show a preconditioner helping (``precond="gmg"`` took 7)
    and cannot detect a regression, which is the whole job of the ``iters`` column
    here. ``test_la_python_api.py`` records the same finding for the Dirichlet
    problem.

    Seeded, so the table is reproducible; the same draw feeds every row and both
    boundary conditions, so the two configurations differ only in the BC.
    """
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _timed(solve: Callable, repeats: int, warmup: int):
    for _ in range(warmup):
        solve()
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        iters, res = solve()
        samples.append(perf_counter() - t0)
    return iters, res, statistics.median(samples)


def bench(n_cell, max_size, bc_name, methods, repeats, warmup, rtol, atol, max_iter):
    periodic = bc_name == "periodic"
    bc = ["periodic" if periodic else "dirichlet"] * 6
    geom, ba, dm = build_mesh(n_cell, max_size, periodic)

    rhs = seeded_rhs(ba, dm)
    sol = blockamr.MultiFab(ba, dm, 1, 1)

    results = []
    for method in methods:
        seam = method in LA_METHODS
        try:
            t0 = perf_counter()
            solve, keepalive = make_solve(
                method, geom, ba, dm, max_size, periodic, bc, rhs, sol, rtol, atol, max_iter
            )
            iters, res = solve()
            setup_ms = 1e3 * (perf_counter() - t0)
            iters, res, solve_s = _timed(solve, repeats, warmup)
            snapshot: np.ndarray | None = solution_to_host(sol)
        except RuntimeError as exc:
            # A refusal is a RESULT — `precond="mlmg"` is listed in the design as
            # reachable through the seam and is not, because the Python Solver
            # binding carries no `precond_mlmg`. Recording it keeps the row in the
            # table instead of quietly shrinking METHODS.
            results.append(
                Result(
                    method=method,
                    bc=bc_name,
                    n_cell=n_cell,
                    seam=seam,
                    iters=-1,
                    setup_ms=float("nan"),
                    solve_ms=float("nan"),
                    mcell_per_s=float("nan"),
                    final_res=float("nan"),
                    status=f"REFUSED: {exc}",
                    snapshot=None,
                )
            )
            continue

        results.append(
            Result(
                method=method,
                bc=bc_name,
                n_cell=n_cell,
                seam=seam,
                iters=iters,
                setup_ms=setup_ms,
                solve_ms=1e3 * solve_s,
                mcell_per_s=n_cell**3 / solve_s / 1e6,
                final_res=res,
                status="ok",
                snapshot=snapshot,
            )
        )
        # The closure and its keepalive hold the only references to this row's
        # AMReX/Ginkgo objects; drop them here so nothing survives to be
        # destroyed after amrex::Finalize (CUDA error 709).
        del solve, keepalive
    return results


def _parity(results, rtol=1e-5, atol=1e-8):
    """Every row solves the same matrix, so the solutions must agree.

    A fast wrong answer must never read as a win. Refused rows have no solution
    to compare and are named in the verdict rather than silently skipped.
    """
    ran = [r for r in results if r.snapshot is not None]
    refused = [r.method for r in results if r.snapshot is None]
    note = f" [no solution: {','.join(refused)}]" if refused else ""
    if len(ran) < 2:
        return "n/a" + note
    ref = ran[0]
    worst = max(float(np.max(np.abs(ref.snapshot - r.snapshot))) for r in ran[1:])
    ok = worst < (atol + rtol * float(np.max(np.abs(ref.snapshot))))
    return f"{'PASS' if ok else 'FAIL'} (max|d|={worst:.2e} vs {ref.method}){note}"


HEADER = (
    "bc",
    "n_cell",
    "method",
    "seam",
    "iters",
    "setup_ms",
    "solve_ms",
    "speedup_vs_mlmg",
    "mcell_per_s",
    "final_res",
    "status",
    "parity",
)


def _print_row(r, speedup, verdict):
    print(
        f"{r.bc:>9} {r.n_cell:>7} {r.method:>11} {'yes' if r.seam else 'no':>4} "
        f"{r.iters:>6} {r.setup_ms:>9.1f} {r.solve_ms:>10.2f} {speedup:>7.2f}x "
        f"{r.mcell_per_s:>9.1f} {r.final_res:>10.2e}  {verdict}"
    )
    if r.status != "ok":
        print(f"{'':>9} {'':>7} {r.method:>11}  -> {r.status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cell", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--max-size", type=int, default=None, help="box size (default: single box)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument(
        "--bc",
        choices=("periodic", "dirichlet", "both"),
        default="both",
        help="boundary condition; 'dirichlet' is the one that exercises the fold and is "
        "the acceptance gate for the future unfold slice",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument(
        "--norm",
        choices=("l2", "linf"),
        default="l2",
        help="convergence norm for the Ginkgo-side rows; 'linf' matches MLMG's criterion "
        "(the mlmg row always uses linf, so this is what makes `iters` comparable)",
    )
    ap.add_argument("--csv", default="bench_solvers2.csv", help="output CSV path")
    args = ap.parse_args()

    global NORM
    NORM = args.norm

    bcs = ("periodic", "dirichlet") if args.bc == "both" else (args.bc,)
    rows = []
    print(
        f"{'bc':>9} {'n_cell':>7} {'method':>11} {'seam':>4} {'iters':>6} {'setup_ms':>9} "
        f"{'solve_ms':>10} {'speedup':>8} {'Mcell/s':>9} {'final_res':>10}  parity"
    )
    with blockamr.runtime():
        for bc_name in bcs:
            for n_cell in args.n_cell:
                max_size = args.max_size if args.max_size is not None else n_cell
                res = bench(
                    n_cell,
                    max_size,
                    bc_name,
                    args.methods,
                    args.repeats,
                    args.warmup,
                    args.rtol,
                    args.atol,
                    args.max_iter,
                )
                verdict = _parity(res)
                base = next((r.solve_ms for r in res if r.method == "mlmg"), None)
                for r in res:
                    speedup = base / r.solve_ms if base else float("nan")
                    _print_row(r, speedup, verdict)
                    rows.append(
                        [
                            r.bc,
                            r.n_cell,
                            r.method,
                            "yes" if r.seam else "no",
                            r.iters,
                            f"{r.setup_ms:.3f}",
                            f"{r.solve_ms:.3f}",
                            f"{speedup:.3f}",
                            f"{r.mcell_per_s:.3f}",
                            f"{r.final_res:.6e}",
                            r.status,
                            verdict,
                        ]
                    )

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
