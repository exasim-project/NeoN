# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Optimising the Kokkos GMG V-cycle against the AMReX one as the orientation point.

The operator bench (``bench_kokkos.py``) compares single kernels. This one compares
a whole solver phase: the V-cycle of ``solvers/gmg_precond.hpp``, run with its AMReX
kernels and with the Kokkos twins in ``bench/gmg_kokkos.hpp``.

Why the V-cycle and not more kernels: it is the launch-bound shape. Per V-cycle it
launches ``(sweeps x 2 colours + 2)`` kernels PER LEVEL, each once per box, with a
ghost exchange between colours. And because production coarsens the BoxArray in
place, the box COUNT is the same on every level while the cell count falls 8x per
level -- so the coarsest level launches as many kernels as the finest for a few
hundred cells. The ``boxes`` and ``cells`` columns below print that hierarchy.

The rows are one baseline and six cumulative changes, so each line is read against
the one above it:

    amrex               the shipped V-cycle, per-box AMReX kernels. Fixed reference:
                        it is deliberately NOT optimised, so every other row is
                        measured against the thing that ships.
    kokkos              the 1:1 Kokkos port, also one launch per box.
    kokkos_fused        the same kernels under one TeamPolicy launch per level.
                        Attacks the per-box launch cost directly.
    kokkos_opt          ... plus the halo exchange, the zero fill and the
                        agglomeration transfers on Kokkos (``halo_kokkos.hpp``),
                        which leaves no AMReX operation inside the timed cycle and
                        so no reason to fence between kernels. The point is not the
                        fences: it is that they forced the host to wait on the
                        device twice per colour sweep, so nothing could overlap.
                        Watch the ``enqueue`` column, which until this row equals
                        ``ms/vcycle`` exactly.
    kokkos_opt+agg      ... plus coarse-grid agglomeration: a coarse level takes a
                        fresh 32-capped decomposition of its domain when that has
                        fewer boxes than coarsening the fine one in place. Depth is
                        pinned to the baseline's, so this is the SAME V-cycle -- the
                        residual is unchanged to the last digit, only the launch
                        count moves.
    kokkos_opt+share    ... plus one face coefficient per direction instead of an
                        upper/lower pair. ux(i+1) is cell i's east coefficient and
                        lx(i+1) is cell i+1's west one -- the same matrix entry for a
                        symmetric operator -- so three of the nine arrays a colour
                        sweep streams were a duplicate of another three. Symmetry is
                        checked at setup, so like agglomeration this is the SAME
                        V-cycle: the residual does not move at all.
    kokkos_opt+fp32     ... plus an fp32 hierarchy, as production's gmg_precision
                        does. Once the launch cost is gone the smoother is bound by
                        memory traffic, and this halves it. The one row that changes
                        arithmetic: r1/r0 moves in the 6th digit, which is fp32
                        rounding and not a different V-cycle.
    +fp32 (deep)        the same, with the depth agglomeration unlocks: big coarse
                        boxes stay coarsenable long after 2^3 boxes stop being, so
                        the hierarchy goes further. This row's r1/r0 legitimately
                        differs -- it is a different (better-coarsened) V-cycle.

``FillBoundary``, ``setVal`` and the agglomeration ``ParallelCopy`` stay AMReX up to
and including ``kokkos_fused``, so those rows pay two host syncs per colour where
AMReX pays one -- their own fence after each kernel, plus a wait on AMReX's stream
after each ``FillBoundary``, since the two runtimes' streams are unordered. That cost
is measured, not hidden, and removing it is what ``kokkos_opt`` is. What stays AMReX
in every row is the untimed hierarchy setup and the residual gate.

The operator is the periodic Helmholtz (phi - laplacian phi) in face-coefficient
form, i.e. the operator ``bench_solvers.py`` hands the persistent solvers.

Usage:
    python bench_gmg_kokkos.py [--csv bench_gmg_kokkos.csv] [--iters 10] [--batches 5]
"""

from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import numpy as np

import blockamr

# (label, backend, agglomerate, pin_depth, fp32, share). pin_depth truncates the
# hierarchy to the baseline's level count, which is what makes a row comparable to the
# baseline rather than merely faster at a different job.
CONFIGS = [
    ("amrex", "amrex", False, True, False, False),
    ("kokkos", "kokkos", False, True, False, False),
    ("kokkos_fused", "kokkos_fused", False, True, False, False),
    ("kokkos_opt", "kokkos_opt", False, True, False, False),
    ("kokkos_opt+agg", "kokkos_opt", True, True, False, False),
    ("kokkos_opt+share", "kokkos_opt", True, True, False, True),
    ("kokkos_opt+fp32", "kokkos_opt", True, True, True, False),
    ("kokkos_opt+fp32+share", "kokkos_opt", True, True, True, True),
    ("+fp32+share deep", "kokkos_opt", True, False, True, True),
]

# The agglomerated coarse box size. 32 not MLMG's 3D default of 8: MLMG agglomerates
# to shrink the number of MPI ranks holding work, which on one GPU would leave the
# box count (and therefore the launch count) untouched.
AGG_GRID_SIZE = 32

# (label, n_cell, max_size). The box count is what this bench is about: max_size
# fixes the box size, so a level's box count is (n_cell / max_size)^3 and stays that
# on every coarser level. 256^3/max_size 32 is 512 boxes down to the bottom, which is
# the regime the coarse levels of any real hierarchy sit in.
CASES = [
    ("64^3/1box", 64, None),
    ("128^3/8box", 128, 64),
    ("256^3/64box", 256, 64),
    ("256^3/512box", 256, 32),
]

# The production V-cycle shape (gmg_precond.hpp defaults): 2 pre + 2 post RB-SOR
# sweeps, and a coarsest level "solved" by sweeping.
PRE_SWEEPS = 2
POST_SWEEPS = 2
COARSEST_SWEEPS = 8
OMEGA = 1.0


def _mesh(n_cell, max_size):
    box = blockamr.Box([0, 0, 0], [n_cell - 1] * 3)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # triply periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(n_cell if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(geom, dm, d, max_size, value):
    dom = geom.domain()
    fb = blockamr.Box(dom.small_end(), dom.big_end())
    fb.surrounding_nodes(d)
    fba = blockamr.BoxArray(fb)
    fba.max_size(max_size)
    mf = blockamr.MultiFab(fba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _rhs(ba, dm, n_cell):
    """A smooth, mean-zero right-hand side: sin(2 pi x) sin(2 pi y) sin(2 pi z)."""
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(0.0)
    dx = 1.0 / n_cell
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        nx, ny, nz = arr.shape[:3]
        x = (lo[0] + np.arange(nx) + 0.5) * dx
        y = (lo[1] + np.arange(ny) + 0.5) * dx
        z = (lo[2] + np.arange(nz) + 0.5) * dx
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
        arr[:, :, :, 0] = np.sin(2 * np.pi * xg) * np.sin(2 * np.pi * yg) * np.sin(2 * np.pi * zg)
        mf.copy_from(mfi, arr)
    return mf


def _run_case(label, n_cell, max_size, iters, batches):
    box_size = n_cell if max_size is None else max_size
    geom, ba, dm = _mesh(n_cell, max_size)
    dx = geom.cell_size()
    alpha = _const_cell(ba, dm, 1.0)
    faces = [_const_face(geom, dm, d, box_size, -1.0 / dx[d] ** 2) for d in range(3)]
    rhs = _rhs(ba, dm, n_cell)

    rows = []
    baseline_levels = 0
    for cfg_label, backend, agglomerate, pin_depth, fp32, share in CONFIGS:
        stats = dict(
            blockamr.bench_gmg_vcycle(
                backend,
                geom,
                rhs,
                alpha,
                faces[0],
                faces[1],
                faces[2],
                pre_sweeps=PRE_SWEEPS,
                post_sweeps=POST_SWEEPS,
                coarsest_sweeps=COARSEST_SWEEPS,
                omega=OMEGA,
                agglomerate=agglomerate,
                agg_grid_size=AGG_GRID_SIZE,
                fp32=fp32,
                share_coeffs=share,
                max_levels=baseline_levels if pin_depth else 0,
                iters=iters,
                batches=batches,
            )
        )
        baseline_levels = baseline_levels or stats["nlevels"]
        rows.append(
            {
                "case": label,
                "backend": cfg_label,
                "nlevels": stats["nlevels"],
                "boxes_per_level": " ".join(str(b) for b in stats["boxes_per_level"]),
                "cells_per_level": " ".join(str(c) for c in stats["cells_per_level"]),
                "ms_min": stats["ms_min"],
                "ms_median": stats["ms_median"],
                "ms_enqueue": stats["ms_enqueue"],
                "resid0": stats["resid0"],
                "resid1": stats["resid1"],
                "shared_coeffs": stats["shared_coeffs"],
            }
        )
    return rows


def _report(rows):
    print(f"\nexecution space: {blockamr.kokkos_execution_space()}")
    for case, _, _ in CASES:
        sel = [r for r in rows if r["case"] == case]
        if not sel:
            continue
        print(f"\n{case}   cells/level {sel[0]['cells_per_level']}")
        print(
            f"  {'config':<22} {'lvls':>4} {'ms/vcycle':>10} {'enqueue':>9} {'r1/r0':>9} "
            f"{'speedup':>9}  boxes/level"
        )
        print("  " + "-" * 81)
        base = next(r["ms_min"] for r in sel if r["backend"] == "amrex")
        for r in sel:
            flag = "" if r["backend"] == "amrex" else f"{base / r['ms_min']:7.2f}x"
            drop = r["resid1"] / r["resid0"] if r["resid0"] > 0 else float("nan")
            print(
                f"  {r['backend']:<22} {r['nlevels']:>4} {r['ms_min']:>10.4f} "
                f"{r['ms_enqueue']:>9.4f} {drop:>9.3e} {flag:>9}  {r['boxes_per_level']}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="bench_gmg_kokkos.csv")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--batches", type=int, default=5)
    args = p.parse_args()

    rows = []

    def run():
        for label, n_cell, max_size in CASES:
            print(f"running {label} ...", flush=True)
            rows.extend(_run_case(label, n_cell, max_size, args.iters, args.batches))

    blockamr.runtime(run)

    _report(rows)
    if rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
