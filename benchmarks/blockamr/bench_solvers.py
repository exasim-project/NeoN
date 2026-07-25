# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""MLMG vs Ginkgo (matrix-free and assembled CSR) on the GPU, over grid size.

Solves the SAME symmetric periodic Helmholtz operator (phi - laplacian phi)
several ways, all on the GPU, and times the solve:

* ``mlmg``       — AMReX geometric multigrid (the reference: what MG buys).
* ``mlmg-nomg``  — the same solver with coarsening disabled
                   (``LPInfo().set_max_coarsening_level(0)``), i.e. a plain
                   Krylov bottom-solve on the finest level, no multigrid.
* ``ginkgo``     — the one-shot matrix-free ``ginkgo_solve`` CG on a
                   ``gko::CudaExecutor``; its mat-vec is the SAME AMReX operator
                   apply MLMG uses. Rebuilds the operator every call.
* ``mf``         — persistent matrix-free ``FaceCoeffSolver`` CG: the operator
                   (OpenFOAM-style alpha + face coefficients) and solver are
                   built ONCE, so each timed solve is only pack -> apply ->
                   unpack. The mat-vec recomputes entries from the face fields.
* ``mf-pmg``     — the same persistent matrix-free CG preconditioned by ONE
                   MLMG V-cycle per iteration (``precond_mlmg``, an MLMG built
                   once on the equivalent MLABecLaplacian — part of setup): the
                   Krylov iteration count then stays ~flat in N like ``mlmg``,
                   while the outer operator stays matrix-free.
* ``mf-gmg``     — the same persistent matrix-free CG preconditioned by the
                   NATIVE geometric-multigrid V-cycle (``precond="gmg"``): the
                   level hierarchy is built once from the face coefficients with
                   AMReX primitives only — no MLLinOp/MLMG anywhere — so each
                   preconditioner apply is a plain V-cycle with none of MLMG's
                   per-solve residual bookkeeping.
* ``mf-gmgk``    — ``mf-gmg`` with the OPTIMISED V-cycle (``precond="gmg_kokkos"``):
                   the same kernels and the same cycle, launched once per level
                   instead of once per box, with the halo exchange and the coarse
                   transfers on Kokkos so no host fence is needed inside a cycle,
                   plus coarse-grid agglomeration. Nothing else differs from
                   ``mf-gmg``, so the pair isolates the V-cycle inside a real solve
                   — and against ``mlmg`` it says whether the optimised native
                   V-cycle beats AMReX's own multigrid.
* ``gmg``        — the NATIVE geometric-multigrid V-cycle run as a STANDALONE
                   stationary solver (``solver="gmg"``): x <- x + V(b - A x)
                   until tolerance, a Richardson iteration exactly like MLMG,
                   with NONE of CG's per-iteration wrapper cost (dots/axpys,
                   separate outer mat-vec, Ginkgo<->AMReX pack/unpack). The whole
                   loop runs on AMReX MultiFabs; Ginkgo is not in the loop (M1).
* ``gmg-ir``     — the SAME native V-cycle driven by Ginkgo's iterative refinement
                   (``solver="ir"``, ``gko::solver::Ir``, relaxation 1.0): the
                   system matrix is the matrix-free ``FaceCoeffOp`` and the inner
                   solver is the generated GMG V-cycle LinOp. Mathematically the
                   same Richardson iteration as ``gmg`` but run THROUGH Ginkgo
                   (Dense pack/unpack + per-iteration LinOp-boundary crossings), so
                   ``gmg`` vs ``gmg-ir`` isolates the cost of the Ginkgo-idiomatic
                   loop over the fully-native one.
* ``csr``        — persistent ``FaceCoeffCsrSolver`` CG: the SAME matrix assembled
                   into a Ginkgo CSR. Unpreconditioned, so ``mf`` vs ``csr`` is a
                   clean apples-to-apples measure of matrix-free (recompute) vs
                   assembled (stream the matrix) — and of the assembly setup cost.

Fairness: identical operator (Helmholtz alpha=beta=a=b=1), identical zero
initial guess, identical relative tolerance, all on one GPU. The convergence
NORM is a fairness axis too: MLMG stops on ||r||_inf <= rtol*||b||_inf and
Ginkgo on the 2-norm, so an iteration count from one is not directly comparable
with the other's. ``--norm linf`` puts the Ginkgo methods on MLMG's criterion
(MLMG's own norm is not configurable), which is the apples-to-apples setting;
``--norm l2`` (the default) keeps each side's native convention. Timing is the
median of ``--repeats`` solves after ``--warmup`` untimed solves. For ``mf``/
``csr`` the one-time operator/solver build (matrix assembly, for ``csr``) is the
reported ``setup_ms`` and is excluded from ``solve_ms``; the one-shot methods
re-pay their build every solve. ``mf`` and ``csr`` use the same unpreconditioned
CG and differ only in how the mat-vec is evaluated.

The ``iters`` column is each solver's NATURAL iteration unit: ``mlmg`` reports
V-cycles (few, ~independent of N), while ``mlmg-nomg`` and ``ginkgo`` report
fine-grid Krylov iterations (which grow with N and always exceed the MG
V-cycle count) — this is the classic multigrid-vs-Krylov contrast. Absolute
wall time is the cross-method comparison. Every AMReX object is created and
released inside ``bench``: if such objects outlive ``blockamr.runtime()`` their
destructors run after amrex::Finalize and abort with CUDA error 709 (context
destroyed).

Run::

    python benchmarks/blockamr/bench_solvers.py --n-cell 32 64 128 --rtol 1e-10
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from time import perf_counter

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import numpy as np

import blockamr

METHODS = (
    "mlmg",
    "mlmg-nomg",
    "ginkgo",
    "mf",
    "mf-pmg",
    "mf-gmg",
    "mf-gmgk",
    "gmg",
    "gmg-ir",
    "csr",
)
# Smoother for the native-GMG preconditioner (mf-gmg); "rbgs" reproduces the
# historical default, "chebyshev" is the M4 polynomial smoother. Set from --gmg-smoother.
GMG_SMOOTHER = "rbgs"
# V-cycle hierarchy precision; "fp64" reproduces the historical default, "fp32" is
# the M5 single-precision V-cycle and "bf16" stores the hierarchy in bfloat16 (still
# computed in fp32). Only mf-gmgk has a bf16 hierarchy, so --gmg-precision bf16
# refuses the methods that would silently run something else. Set from
# --gmg-precision.
GMG_PRECISION = "fp64"
# Convergence norm for the Ginkgo-side methods; "l2" reproduces the historical
# default, "linf" is MLMG's own criterion (||r||_inf <= rtol*||b||_inf). The mlmg
# rows ALWAYS stop on linf -- it is not configurable in MLMG -- so --norm linf is
# what makes the iteration counts in this table comparable across methods rather
# than each method's own convention. Set from --norm.
NORM = "l2"
# MLMG's own V-cycle shape, so the REFERENCE solver can be tuned on the same axes
# we tune ourselves on. AMReX defaults: nu1 = nu2 = 2 (the same 2+2 the native GMG
# defaults to), nub = 0, semicoarsening off. Set from --mlmg-*.
MLMG_PRE_SMOOTH = 2
MLMG_POST_SMOOTH = 2
MLMG_BOTTOM_SMOOTH = 0
# Max semicoarsening levels; 0 = off (AMReX's default).
MLMG_SEMICOARSENING = 0
# Target box size for level 0 of the mf-gmgk hierarchy; 0 keeps the caller's boxes.
# Set from --gmg-agg-l0-size.
GMG_AGG_L0_SIZE = 0
# Persistent solvers built once and reused (per-solve = pack/apply/unpack only).
PERSISTENT = {
    "mf": "FaceCoeffSolver",
    "mf-pmg": "FaceCoeffSolver",
    "mf-gmg": "FaceCoeffSolver",
    "mf-gmgk": "FaceCoeffSolver",
    "gmg": "FaceCoeffSolver",
    "gmg-ir": "FaceCoeffSolver",
    "csr": "FaceCoeffCsrSolver",
}


# ---------------------------------------------------------------------------
# Mesh + field helpers (single box -> face fabs align 1:1 with the cell fab;
# same decomposition for every method).
# ---------------------------------------------------------------------------
def build_mesh(n_cell: int, max_size: int):
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # triply periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def fill_cell(mf, dx, fn):
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        nx, ny, nz = arr.shape[:3]
        x = (lo[0] + np.arange(nx) + 0.5) * dx[0]
        y = (lo[1] + np.arange(ny) + 0.5) * dx[1]
        z = (lo[2] + np.arange(nz) + 0.5) * dx[2]
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
        arr[:, :, :, 0] = fn(xg, yg, zg)
        mf.copy_from(mfi, arr)


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


def tune_mlmg(mlmg):
    """Apply the MLMG V-cycle knobs from --mlmg-*.

    These are the direct counterparts of the gmg_* knobs the native side is tuned
    on (nu1/nu2/nub against pre/post/coarsest sweeps). Without them the table
    compares a tuned solver against a stock one, which flatters us for free.
    """
    mlmg.set_pre_smooth(MLMG_PRE_SMOOTH)
    mlmg.set_post_smooth(MLMG_POST_SMOOTH)
    mlmg.set_bottom_smooth(MLMG_BOTTOM_SMOOTH)
    return mlmg


def make_abec(geom, ba, dm, max_size, max_coarsen=None):
    """Periodic Helmholtz (phi - laplacian phi): MLABecLaplacian, alpha=beta=a=b=1."""
    info = blockamr.LPInfo()
    if max_coarsen is not None:
        info.set_max_coarsening_level(max_coarsen)
    # Semicoarsening is MLMG's answer to anisotropic cells: coarsen only the
    # strongly-coupled direction instead of all three. Off unless asked for, which
    # is AMReX's own default; it does nothing on the isotropic cells this bench uses
    # but has to be reachable for the anisotropic comparison to be honest.
    if MLMG_SEMICOARSENING > 0:
        info.set_semicoarsening(True)
        info.set_max_semicoarsening_level(MLMG_SEMICOARSENING)
    abec = blockamr.MLABecLaplacian(geom, ba, dm, info)
    abec.set_domain_bc([blockamr.LinOpBCType.Periodic] * 3, [blockamr.LinOpBCType.Periodic] * 3)
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)
    abec.set_a_coeffs(0, const_cell(ba, dm, 1.0))
    abec.set_b_coeffs(
        0,
        const_face(geom, dm, 0, max_size, 1.0),
        const_face(geom, dm, 1, max_size, 1.0),
        const_face(geom, dm, 2, max_size, 1.0),
    )
    return abec


def solution_to_host(mf):
    return np.concatenate(
        [mf.copy_to_host(mfi)[:, :, :, 0].ravel() for mfi in blockamr.MFIterator(mf)]
    )


# ---------------------------------------------------------------------------
# Face-coefficient form of the SAME periodic Helmholtz (phi - lap phi): diagonal
# source alpha=1, symmetric face coefficients -1/dx^2. Handed to the persistent
# matrix-free / CSR solvers exactly as an external discretiser would.
# ---------------------------------------------------------------------------
def build_face_coeffs(geom, ba, dm, max_size):
    dx = geom.cell_size()
    alpha = const_cell(ba, dm, 1.0)
    fx = const_face(geom, dm, 0, max_size, -1.0 / dx[0] ** 2)
    fy = const_face(geom, dm, 1, max_size, -1.0 / dx[1] ** 2)
    fz = const_face(geom, dm, 2, max_size, -1.0 / dx[2] ** 2)
    return alpha, fx, fy, fz


def make_persistent(method, geom, ba, dm, max_size, rtol, max_iter):
    """Build a persistent solver ONCE. It keeps the coefficient fields alive."""
    alpha, fx, fy, fz = build_face_coeffs(geom, ba, dm, max_size)
    cls = getattr(blockamr, PERSISTENT[method])
    kwargs = {}
    if method == "mf-pmg":
        # MLMG preconditioner on the equivalent assembled operator: one V-cycle
        # per Krylov iteration. Built here, so it counts as setup; the solver's
        # keep_alive ties its lifetime to the returned object.
        mlmg = tune_mlmg(blockamr.MLMG(make_abec(geom, ba, dm, max_size)))
        mlmg.set_verbose(0)
        kwargs = {"precond_mlmg": mlmg, "precond_cycles": 1}
    elif method == "mf-gmg":
        # Native matrix-free geometric multigrid: the whole hierarchy is built
        # inside the solver constructor (setup-timed), from the face
        # coefficients alone — no MLMG object involved.
        kwargs = {
            "precond": "gmg",
            "precond_cycles": 1,
            "gmg_smoother": GMG_SMOOTHER,
            "gmg_precision": GMG_PRECISION,
        }
    elif method == "mf-gmgk":
        # The same V-cycle as mf-gmg, under the optimised Kokkos launchers
        # (precond="gmg_kokkos"): fused one-launch-per-level kernels, the halo
        # exchange and coarse transfers on Kokkos so the cycle needs no host fence,
        # and coarse-grid agglomeration. Everything else -- operator, Krylov, sweep
        # counts, tolerance -- is mf-gmg's, so mf-gmgk / mf-gmg isolates the V-cycle.
        kwargs = {
            "precond": "gmg_kokkos",
            "precond_cycles": 1,
            "gmg_smoother": GMG_SMOOTHER,
            "gmg_precision": GMG_PRECISION,
            "gmg_agg_l0_size": GMG_AGG_L0_SIZE,
        }
    elif method in ("gmg", "gmg-ir"):
        # Native stationary GMG solver (solver="gmg") or its Ginkgo iterative-
        # refinement twin (solver="ir"): both are x <- x + V(b - A x), a Richardson
        # iteration on the same V-cycle. "gmg" runs the loop natively on AMReX
        # MultiFabs (no Ginkgo in the loop, M1); "ir" runs it through Ginkgo's Ir
        # (Dense pack/unpack + LinOp-boundary crossings). Both need the same
        # accurate coarsest-grid solve, so coarsest sweeps are raised identically
        # (the bottom is <=4^3, so this is cheap at scale).
        kwargs = {
            "precond_cycles": 1,
            "gmg_smoother": GMG_SMOOTHER,
            "gmg_precision": GMG_PRECISION,
            "gmg_coarsest_sweeps": 160 if GMG_SMOOTHER == "chebyshev" else 100,
        }
    solver = {"gmg": "gmg", "gmg-ir": "ir"}.get(method, "cg")
    return cls(
        alpha=alpha,
        ux=fx,
        lx=fx,
        uy=fy,
        ly=fy,
        uz=fz,
        lz=fz,
        geom=geom,
        executor="cuda",
        solver=solver,
        max_iter=max_iter,
        rtol=rtol,
        norm=NORM,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# One solve of a given method (fresh operator + zero initial guess each call).
# Returns (num_iters, final_res). The returned residual is a host scalar, which
# forces device completion -> the enclosing perf_counter span is a full solve.
# ---------------------------------------------------------------------------
def solve_once(method, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter):
    sol.set_val(0.0)
    if method == "ginkgo":
        stats = blockamr.ginkgo_solve(
            make_abec(geom, ba, dm, max_size),
            sol,
            rhs,
            max_iter=max_iter,
            rtol=rtol,
            sign=1.0,  # MLABecLaplacian is already SPD
            executor="cuda",
        )
        return stats["num_iters"], stats["res_norm"]

    max_coarsen = 0 if method == "mlmg-nomg" else None
    mlmg = tune_mlmg(blockamr.MLMG(make_abec(geom, ba, dm, max_size, max_coarsen)))
    mlmg.set_verbose(0)
    mlmg.set_max_iter(max_iter)
    mlmg.set_bottom_solver("cg")  # SPD
    mlmg.set_bottom_max_iter(max_iter)
    res = mlmg.solve(sol, rhs, rtol, atol)
    # Report each solver's NATURAL iteration unit. True multigrid converges in a
    # few V-cycles, ~independent of N (get_num_iters). Without coarsening the
    # work is the fine-grid bottom Krylov solve, whose total iteration count
    # (get_num_cg_iters, summed over outer cycles) grows with N like the
    # matrix-free ginkgo CG -- and always exceeds the MG V-cycle count.
    iters = mlmg.get_num_cg_iters() if method == "mlmg-nomg" else mlmg.get_num_iters()
    return iters, res


def do_solve(method, obj, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter):
    """One solve from a zero initial guess. Persistent solvers reuse `obj`."""
    if obj is not None:  # persistent (mf / csr)
        sol.set_val(0.0)
        st = obj.solve(rhs, sol)
        return st["num_iters"], st["res_norm"]
    return solve_once(method, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter)


@dataclass
class Result:
    method: str
    n_cell: int
    iters: int
    setup_ms: float
    solve_ms: float
    mcell_per_s: float
    final_res: float
    snapshot: np.ndarray


def bench(n_cell, max_size, methods, repeats, warmup, rtol, atol, max_iter):
    geom, ba, dm = build_mesh(n_cell, max_size)
    dx = geom.cell_size()
    pi = math.pi

    def rhs_fn(x, y, z):
        # Several periodic modes so an unpreconditioned Krylov must iterate
        # across the spectrum (a single eigenmode would converge trivially).
        return (
            np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z)
            + np.sin(4 * pi * x) * np.sin(2 * pi * y)
            + np.cos(2 * pi * x) * np.cos(4 * pi * z)
            + 0.5
        )

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    fill_cell(rhs, dx, rhs_fn)
    sol = blockamr.MultiFab(ba, dm, 1, 1)

    results = []
    for method in methods:
        # A persistent solver builds the operator+solver ONCE (that build is the
        # setup); each timed solve is then only pack -> apply -> unpack. A
        # non-persistent method rebuilds per call, so its setup is the first
        # solve and every repeat re-pays the build.
        obj = None
        t0 = perf_counter()
        if method in PERSISTENT:
            obj = make_persistent(method, geom, ba, dm, max_size, rtol, max_iter)
        iters, res = do_solve(method, obj, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter)
        setup_ms = 1e3 * (perf_counter() - t0)

        for _ in range(warmup):
            do_solve(method, obj, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter)

        samples = []
        for _ in range(repeats):
            t0 = perf_counter()
            iters, res = do_solve(
                method, obj, geom, ba, dm, max_size, rhs, sol, rtol, atol, max_iter
            )
            samples.append(perf_counter() - t0)

        solve_s = statistics.median(samples)
        results.append(
            Result(
                method=method,
                n_cell=n_cell,
                iters=iters,
                setup_ms=setup_ms,
                solve_ms=1e3 * solve_s,
                mcell_per_s=n_cell**3 / solve_s / 1e6,
                final_res=res,
                snapshot=solution_to_host(sol),
            )
        )
    return results


def _parity(results, rtol=1e-5, atol=1e-8):
    """All methods solve the same matrix, so solutions must agree."""
    if len(results) < 2:
        return "n/a"
    ref = results[0]
    worst = 0.0
    for r in results[1:]:
        worst = max(worst, float(np.max(np.abs(ref.snapshot - r.snapshot))))
    ok = worst < (atol + rtol * float(np.max(np.abs(ref.snapshot))))
    return f"{'OK' if ok else 'FAIL'} (max|Δ|={worst:.2e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cell", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--max-size", type=int, default=None, help="box size (default: single box)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument(
        "--norm",
        choices=("l2", "linf"),
        default="l2",
        help="convergence norm for the Ginkgo methods; 'linf' matches MLMG's criterion "
        "(mlmg always uses linf, so this is what makes iters comparable)",
    )
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--csv", default="bench_solvers.csv", help="output CSV path")
    ap.add_argument(
        "--gmg-smoother",
        default="rbgs",
        choices=("rbgs", "chebyshev"),
        help="smoother for the mf-gmg native-GMG preconditioner (default rbgs)",
    )
    ap.add_argument(
        "--gmg-agg-l0-size",
        type=int,
        default=0,
        help="target box size for level 0 of the mf-gmgk hierarchy (0 = the caller's boxes)",
    )
    ap.add_argument(
        "--gmg-precision",
        default="fp64",
        choices=("fp64", "fp32", "bf16"),
        help="V-cycle hierarchy precision (default fp64); bf16 needs --methods mf-gmgk",
    )
    # MLMG's own V-cycle shape. Defaults are AMReX's, so leaving these alone
    # reproduces the historical table; they exist so the REFERENCE can be tuned on
    # the same axes as ours instead of being compared at stock settings.
    ap.add_argument(
        "--mlmg-pre-smooth", type=int, default=2,
        help="MLMG nu1, pre-smoothing sweeps (AMReX default 2)",
    )
    ap.add_argument(
        "--mlmg-post-smooth", type=int, default=2,
        help="MLMG nu2, post-smoothing sweeps (AMReX default 2)",
    )
    ap.add_argument(
        "--mlmg-bottom-smooth", type=int, default=0,
        help="MLMG nub, extra smoothing after the bottom solve (AMReX default 0)",
    )
    ap.add_argument(
        "--mlmg-semicoarsening", type=int, default=0,
        help="MLMG max semicoarsening levels (0 = off, AMReX's default). Coarsens only "
        "the strongly-coupled direction; the designed answer to anisotropic cells",
    )
    args = ap.parse_args()

    # bf16 exists for precond="gmg_kokkos" alone. Rather than let the other GMG rows
    # quietly run fp64 under a bf16 heading -- which is exactly what makes a
    # comparison table lie -- name them and stop.
    if args.gmg_precision == "bf16":
        no_bf16 = sorted({"mf-gmg", "gmg", "gmg-ir"} & set(args.methods))
        if no_bf16:
            ap.error(
                "--gmg-precision bf16 is implemented for mf-gmgk only; "
                f"drop {', '.join(no_bf16)} from --methods"
            )
    global GMG_SMOOTHER, GMG_PRECISION, NORM, GMG_AGG_L0_SIZE
    global MLMG_PRE_SMOOTH, MLMG_POST_SMOOTH, MLMG_BOTTOM_SMOOTH, MLMG_SEMICOARSENING
    GMG_SMOOTHER = args.gmg_smoother
    GMG_PRECISION = args.gmg_precision
    NORM = args.norm
    GMG_AGG_L0_SIZE = args.gmg_agg_l0_size
    MLMG_PRE_SMOOTH = args.mlmg_pre_smooth
    MLMG_POST_SMOOTH = args.mlmg_post_smooth
    MLMG_BOTTOM_SMOOTH = args.mlmg_bottom_smooth
    MLMG_SEMICOARSENING = args.mlmg_semicoarsening

    header = [
        "n_cell",
        "method",
        "iters",
        "setup_ms",
        "solve_ms",
        "speedup_vs_mlmg",
        "mcell_per_s",
        "final_res",
        "parity",
    ]
    rows = []
    print(
        f"{'n_cell':>7} {'method':>11} {'iters':>7} {'setup_ms':>9} "
        f"{'solve_ms':>10} {'speedup':>8} {'Mcell/s':>9} {'final_res':>10}  parity"
    )
    with blockamr.runtime():
        for n_cell in args.n_cell:
            max_size = args.max_size if args.max_size is not None else n_cell
            res = bench(
                n_cell,
                max_size,
                args.methods,
                args.repeats,
                args.warmup,
                args.rtol,
                args.atol,
                args.max_iter,
            )
            verdict = _parity(res)
            # mlmg is the baseline; speedup = t_mlmg / t_method (>1 faster, <1 slower).
            base = next((r.solve_ms for r in res if r.method == "mlmg"), None)
            for r in res:
                speedup = base / r.solve_ms if base is not None else float("nan")
                print(
                    f"{r.n_cell:>7} {r.method:>11} {r.iters:>7} {r.setup_ms:>9.1f} "
                    f"{r.solve_ms:>10.2f} {speedup:>7.2f}x {r.mcell_per_s:>9.1f} "
                    f"{r.final_res:>10.2e}  {verdict}"
                )
                rows.append(
                    [
                        r.n_cell,
                        r.method,
                        r.iters,
                        f"{r.setup_ms:.3f}",
                        f"{r.solve_ms:.3f}",
                        f"{speedup:.3f}",
                        f"{r.mcell_per_s:.3f}",
                        f"{r.final_res:.6e}",
                        verdict,
                    ]
                )

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
