# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Kokkos vs AMReX on the native GMG V-cycle, over the same level hierarchy.

The operator bench (``bench_kokkos.py``) compares single kernels. This one compares
a whole solver phase: the V-cycle of ``solvers/gmg_precond.hpp``, run once with its
AMReX kernels and once with the Kokkos twins in ``bench/gmg_kokkos.hpp``. Same
hierarchy, same sweep counts, same order of operations -- only the launcher differs.

Why the V-cycle and not more kernels: it is the launch-bound shape. Per V-cycle it
launches ``(sweeps x 2 colours + 2)`` kernels PER LEVEL, each once per box, with a
ghost exchange between colours. And because the hierarchy coarsens the BoxArray in
place (no agglomeration), the box COUNT is the same on every level while the cell
count falls 8x per level -- so the coarsest level launches as many kernels as the
finest for a few hundred cells. The ``boxes`` and ``cells`` columns below print that
hierarchy per case.

What is NOT ported, and is AMReX in both columns: ``FillBoundary`` (a halo exchange,
not a cell loop), ``setVal``, and the untimed hierarchy setup. The Kokkos column
therefore pays two host syncs per colour where AMReX pays one -- its own fence after
each kernel, plus a wait on AMReX's stream after each ``FillBoundary``, since the two
runtimes' streams are unordered. That cost is part of what is measured, not hidden.

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

BACKENDS = ["amrex", "kokkos"]

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
    for backend in BACKENDS:
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
                iters=iters,
                batches=batches,
            )
        )
        rows.append(
            {
                "case": label,
                "backend": backend,
                "nlevels": stats["nlevels"],
                "boxes_per_level": " ".join(str(b) for b in stats["boxes_per_level"]),
                "cells_per_level": " ".join(str(c) for c in stats["cells_per_level"]),
                "ms_min": stats["ms_min"],
                "ms_median": stats["ms_median"],
                "ms_enqueue": stats["ms_enqueue"],
                "resid0": stats["resid0"],
                "resid1": stats["resid1"],
            }
        )
    return rows


def _report(rows):
    print(f"\nexecution space: {blockamr.kokkos_execution_space()}")
    print(
        f"{'case':<14} {'backend':<8} {'lvls':>4} {'ms/vcycle':>10} {'enqueue':>9} "
        f"{'r1/r0':>8} {'vs amrex':>9}"
    )
    print("-" * 68)
    for case, _, _ in CASES:
        sel = [r for r in rows if r["case"] == case]
        if not sel:
            continue
        base = next(r["ms_min"] for r in sel if r["backend"] == "amrex")
        for r in sel:
            ratio = r["ms_min"] / base
            flag = "" if r["backend"] == "amrex" else f"{ratio:8.2f}x"
            drop = r["resid1"] / r["resid0"] if r["resid0"] > 0 else float("nan")
            print(
                f"{r['case']:<14} {r['backend']:<8} {r['nlevels']:>4} "
                f"{r['ms_min']:>10.4f} {r['ms_enqueue']:>9.4f} {drop:>8.2e} {flag:>9}"
            )
        print(f"{'':<14} boxes/level {sel[0]['boxes_per_level']}")
        print(f"{'':<14} cells/level {sel[0]['cells_per_level']}")
        print()


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
