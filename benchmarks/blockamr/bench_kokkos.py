# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Kokkos vs AMReX on the same cell kernels, over the same MultiFab memory.

Question: would replacing amrex::ParallelFor with Kokkos cost anything? Three
kernels (axpy, 7-point Laplacian, VanLeer divergence) each run through seven
launchers. The kernel BODY is one templated function shared by all of them, so what
varies is the launch machinery, not the arithmetic.

Per box -- amrex::ParallelFor; Kokkos MDRangePolicy<Rank<3>>; Kokkos RangePolicy
with manual ijk decomposition (AMReX's own scheme, and the only form
NeoN::parallelFor can express today); MDRangePolicy with AMReX's Array4 accessor
(isolates launcher from accessor); and MDRangePolicy round-robined over as many
Kokkos streams as AMReX uses.

Fused, one launch for all boxes -- amrex::ParallelFor(mf, f), AMReX's own fused
path, against Kokkos TeamPolicy over the same block decomposition and the same
cached BoxIndexer table. These exist because the per-box columns pay a launch per
box, and the multi-box rows below showed that is where a gap appears.

The multi-box rows matter as much as the large single-box rows: launches are per
box, so that is where dispatch and launch overhead show up.

Caveat on flags: the bench kernels compile in a non-RDC object library (Kokkos'
desul atomics reject AMReX's -rdc=true), so the AMReX kernels here are non-RDC
while production AMReX kernels are RDC. Both backends get identical flags, which
is what makes the comparison fair; it is not a prediction of production AMReX.

Usage:
    python bench_kokkos.py [--csv bench_kokkos.csv] [--iters 50] [--batches 5]
"""

import argparse
import csv
import sys

import numpy as np

import blockamr

BACKENDS = [
    "amrex",
    "kokkos_md",
    "kokkos_flat",
    "kokkos_md_a4",
    "kokkos_stream",
    "amrex_fused",
    "kokkos_team",
]
KERNELS = ["axpy", "laplacian", "vanleer"]

# (label, shape, max_size).
#
# CACHE WARNING when reading the GB/s column: this GPU has a 36 MB L2, so at 128^3
# and below both arrays (2 x 16.8 MB) are L2-resident and the rate reported is L2
# bandwidth, several times the ~504 GB/s DRAM peak. Only the 256^3 rows (134 MB per
# array) measure DRAM traffic. Compare backends within a row; do not read the
# smaller rows as achieved memory bandwidth.
#
# The multibox rows exist because launches are per box: 128^3/8box isolates
# per-launch cost while each box is still L2-resident, and the 256^3 splits ask
# whether that cost still matters once every box is DRAM-bound. The two 512-box
# rows are the launch-dominated end: 512 boxes of 8^3 is 512 launches over 262k
# cells total, which no per-box backend can hide, and is the regime a GMG
# hierarchy's coarse levels actually live in.
CASES = [
    ("32^3", (32, 32, 32), None),
    ("64^3", (64, 64, 64), None),
    ("128^3", (128, 128, 128), None),
    ("256^3", (256, 256, 256), None),
    ("128^3/8box", (128, 128, 128), 64),
    ("256^3/8box", (256, 256, 256), 128),
    ("256^3/64box", (256, 256, 256), 64),
    ("64^3/512box", (64, 64, 64), 8),
    ("256^3/512box", (256, 256, 256), 32),
]


def _mesh(shape, max_size):
    box = blockamr.Box([0, 0, 0], [s - 1 for s in shape])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(shape) if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _scatter(mf, values):
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1]
        mf.copy_from(mfi, arr)
    return mf


def _cell_mf(ba, dm, geom, values, nghost):
    mf = blockamr.MultiFab(ba, dm, 1, nghost)
    mf.set_val(0.0)
    _scatter(mf, values)
    if nghost:
        mf.fill_boundary(geom)
    return mf


def _face_mf(ba, dm, d, values):
    typ = [0, 0, 0]
    typ[d] = 1
    fba = blockamr.convert_ba(ba, blockamr.IntVect(*typ))
    mf = blockamr.MultiFab(fba, dm, 1, 0)
    mf.set_val(0.0)
    _scatter(mf, values)
    return mf


def _run_case(label, shape, max_size, iters, batches):
    geom, ba, dm = _mesh(shape, max_size)
    dx = tuple(1.0 / s for s in shape)
    rng = np.random.default_rng(0)
    phi = rng.random(shape)

    rows = []
    for kernel in KERNELS:
        info = blockamr.bench_operator_info(f"{kernel}/amrex")
        in_mf = _cell_mf(ba, dm, geom, phi, info["nghost"])
        out_mf = _cell_mf(ba, dm, geom, np.zeros(shape), 0)
        kwargs = {}
        if info["needs_faces"]:
            faces = [
                rng.random(tuple(s + (1 if a == d else 0) for a, s in enumerate(shape))) - 0.5
                for d in range(3)
            ]
            fmfs = [_face_mf(ba, dm, d, faces[d]) for d in range(3)]
            kwargs = {"fx": fmfs[0], "fy": fmfs[1], "fz": fmfs[2]}

        for backend in BACKENDS:
            stats = dict(
                blockamr.bench_operator(
                    f"{kernel}/{backend}",
                    out_mf,
                    in_mf,
                    dx=dx[0],
                    dy=dx[1],
                    dz=dx[2],
                    iters=iters,
                    batches=batches,
                    **kwargs,
                )
            )
            rows.append(
                {
                    "case": label,
                    "kernel": kernel,
                    "backend": backend,
                    "nboxes": stats["nboxes"],
                    "ncells": stats["ncells"],
                    "ms_min": stats["ms_min"],
                    "ms_median": stats["ms_median"],
                    "gb_per_s": stats["gb_per_s"],
                }
            )
    return rows


def _report(rows):
    """One block per (case, kernel), AMReX first, with each Kokkos launcher's ratio."""
    print(f"\nexecution space: {blockamr.kokkos_execution_space()}")
    print(
        f"{'case':<12} {'kernel':<10} {'backend':<12} {'boxes':>5} "
        f"{'ms/apply':>10} {'GB/s':>8} {'vs amrex':>9}"
    )
    print("-" * 72)
    for case, _, _ in CASES:
        for kernel in KERNELS:
            sel = [r for r in rows if r["case"] == case and r["kernel"] == kernel]
            if not sel:
                continue
            base = next(r["ms_min"] for r in sel if r["backend"] == "amrex")
            for r in sel:
                ratio = r["ms_min"] / base
                flag = "" if r["backend"] == "amrex" else f"{ratio:8.2f}x"
                print(
                    f"{r['case']:<12} {r['kernel']:<10} {r['backend']:<12} "
                    f"{r['nboxes']:>5} {r['ms_min']:>10.4f} {r['gb_per_s']:>8.1f} {flag:>9}"
                )
            print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="bench_kokkos.csv")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument(
        "--max-cells", type=int, default=None, help="skip cases larger than this many cells"
    )
    args = p.parse_args()

    rows = []

    def run():
        for label, shape, max_size in CASES:
            ncells = shape[0] * shape[1] * shape[2]
            if args.max_cells is not None and ncells > args.max_cells:
                print(f"skipping {label} ({ncells} cells > --max-cells)", file=sys.stderr)
                continue
            print(f"running {label} ...", file=sys.stderr)
            rows.extend(_run_case(label, shape, max_size, args.iters, args.batches))

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
