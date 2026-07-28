# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Run the backend sweep, extract performance data to CSV, and plot it.

Imports the benchmark harness, runs a (n_cell × max_size) grid for both
backends, writes ``results.csv``, and renders ``backend_perf.png``:

    * per-step time vs number of boxes (box-count scaling)
    * throughput (Mcell/s) vs number of boxes

Usage::

    python plot_backends.py [--steps 30] [--out backend_perf.png]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import blockamr
import bench_backends as bb

# 2D sweep: vary cell count (n_cell) AND box size (max_size). max_size must
# divide n_cell; #boxes = (n_cell / max_size)**3. Overridable on the CLI.
N_CELLS = (32, 64, 128)
MAX_SIZES = (16, 32, 64)
BACKENDS = ("jax", "cpp")
# One colour per max_size (box size); backends live in separate panels.
MS_COLORS = {16: "#d1495b", 32: "#edae49", 64: "#2e86ab", 128: "#66a182", 256: "#8338ec"}
HERE = Path(__file__).parent


def collect(steps: int, warmup: int, seed: int, n_cells, max_sizes) -> list[dict]:
    sweep = [(n, ms) for n in n_cells for ms in max_sizes if ms <= n and n % ms == 0]
    rows: list[dict] = []
    with blockamr.runtime():
        for n_cell, max_size in sweep:
            for r in bb.bench(n_cell, max_size, BACKENDS, steps, warmup, seed):
                rows.append(
                    {
                        "n_cell": r.n_cell,
                        "max_size": r.max_size,
                        "boxes": (r.n_cell // r.max_size) ** 3,
                        "cells": r.n_cell ** 3,
                        "backend": r.backend,
                        "compile_ms": r.compile_ms,
                        "per_step_ms": r.per_step_ms,
                        "mcell_per_s": r.mcell_updates_per_s,
                    }
                )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], out: Path) -> None:
    """2×2: rows = {per-step time, throughput}, cols = {jax, cpp}.

    Within each panel, x = n_cell (cell count), one line per max_size (box size)
    — so both swept axes are visible at once.
    """
    metrics = [
        ("per_step_ms", "per-step wall time [ms]", "log"),
        ("mcell_per_s", "throughput [Mcell-updates / s]", "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    n_ticks = sorted({r["n_cell"] for r in rows})

    for i, (key, ylabel, yscale) in enumerate(metrics):
        for j, backend in enumerate(BACKENDS):
            ax = axes[i][j]
            for max_size in MAX_SIZES:
                pts = sorted(
                    (r for r in rows if r["backend"] == backend and r["max_size"] == max_size),
                    key=lambda r: r["n_cell"],
                )
                if not pts:
                    continue
                ax.plot(
                    [r["n_cell"] for r in pts], [r[key] for r in pts],
                    "o-", color=MS_COLORS[max_size], lw=2, ms=7,
                    label=f"max_size={max_size}",
                )
            ax.set(xscale="log", yscale=yscale, title=f"{backend}")
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.set_xticks(n_ticks)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("n_cell (cells per dimension)")
            ax.legend(fontsize=8)

    # Share y within each metric row so jax/cpp magnitudes compare directly.
    for i in range(2):
        lo = min(axes[i][j].get_ylim()[0] for j in range(2))
        hi = max(axes[i][j].get_ylim()[1] for j in range(2))
        for j in range(2):
            axes[i][j].set_ylim(lo, hi)

    fig.suptitle(
        "blockAMR explicit backend sweep — vary cell count × box size\n"
        "jax (per-box host Python) vs cpp (device ParallelFor)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    global BACKENDS, MAX_SIZES
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--n-cells", type=int, nargs="+", default=list(N_CELLS))
    ap.add_argument("--max-sizes", type=int, nargs="+", default=list(MAX_SIZES))
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--csv", type=Path, default=HERE / "results.csv")
    ap.add_argument("--out", type=Path, default=HERE / "backend_perf.png")
    args = ap.parse_args()

    BACKENDS = tuple(args.backends)
    MAX_SIZES = tuple(args.max_sizes)
    rows = collect(args.steps, args.warmup, args.seed, args.n_cells, args.max_sizes)
    write_csv(rows, args.csv)
    print(f"wrote {args.csv} ({len(rows)} rows)")
    plot(rows, args.out)


if __name__ == "__main__":
    main()
