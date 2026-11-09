# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Generate figures/neon_performance.png of the JOSS paper.

The data is produced by the ``momentum`` benchmark of NeoFOAM
(benchmarks/momentum.cpp, registered via ``NeoN_benchmark``), which assembles
``imp::ddt(U) + imp::div(phi, U) - imp::laplacian(nu, U)`` on a unit cube. The
benchmark suite (benchmarks/benchmarkSuite/createStudies.py) writes one
momentum.csv per run, see runPerformance.sh. The csv files used for the figure
are stored in figures/data.

Left:  assembly time per cell over the number of cells for every GPU file and
       for the CPUExecutor using 1 and --mt-threads threads.
Right: CPUExecutor speedup relative to a single thread.

Usage:
    cd doc/joss/figures
    python3 ../scripts/plotPerformance.py \\
        --gpu A100=data/momentum_nvidia_a100.1.csv --gpu H200=data/momentum_nvidia_h200.1.csv \\
        --cpu data/momentum_nvidia_h200.{1,2,4,8,16,32}.csv \\
        --output neon_performance.png

CPU files must be named <prefix>.<threads>.csv.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load(path, section):
    df = pd.read_csv(path)
    df = df[(df["MeshType"] == "3DCube") & (df["section2"] == section)].copy()
    df["Resolution"] = df["Resolution"].str[1:].astype(int)
    df["Cells"] = df["Resolution"] ** 3
    df["Time/Cell"] = df["avg_runtime"] / df["Cells"]
    return df


def threads_of(path):
    match = re.search(r"\.(\d+)\.csv$", str(path))
    if not match:
        raise ValueError(f"cannot read the thread count from {path}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--cpu", nargs="+", required=True, type=Path, metavar="CSV")
    parser.add_argument("--mt-threads", type=int, default=16, help="threads of the multi-core curve")
    parser.add_argument("--min-res", type=int, default=75)
    parser.add_argument("--max-res", type=int, default=275)
    parser.add_argument("--section", default="without RHS")
    parser.add_argument("--output", type=Path, default=Path("neon_performance.png"))
    args = parser.parse_args()

    def in_range(df):
        return df[df["Resolution"].between(args.min_res, args.max_res)]

    cpu = []
    for path in args.cpu:
        df = load(path, args.section)
        df = df[df["benchmark_name"] == "CPUExecutor"]
        df["Threads"] = threads_of(path)
        cpu.append(df)
    cpu = in_range(pd.concat(cpu))

    curves = []
    for spec in args.gpu:
        label, path = spec.split("=", 1)
        df = load(path, args.section)
        df = df[df["benchmark_name"] == "GPUExecutor"]
        curves.append(df.assign(Label=label))
    curves.append(cpu[cpu["Threads"] == args.mt_threads].assign(Label=f"{args.mt_threads} Threads"))
    curves.append(cpu[cpu["Threads"] == 1].assign(Label="Single Core"))
    curves = in_range(pd.concat(curves))

    baseline = cpu[cpu["Threads"] == 1].set_index("Resolution")["avg_runtime"]
    cpu["Speedup [-]"] = cpu["Resolution"].map(baseline) / cpu["avg_runtime"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(
        data=curves, x="Cells", y="Time/Cell", hue="Label", style="benchmark_name",
        markers=["o", "o"], dashes=["", (4, 1.5, 1, 1.5)], ax=left,
    )
    left.set(xscale="log", yscale="log", ylabel="Assembly Time/Cell [ns]")
    handles, labels = left.get_legend_handles_labels()
    labels = ["GPU" if lbl == "Label" else lbl for lbl in labels]
    left.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)

    sns.barplot(
        data=cpu, x="Threads", y="Speedup [-]", hue="Resolution",
        palette=(sns.color_palette("husl", 6) * 2)[: cpu["Resolution"].nunique()], ax=right,
    )
    right.legend(title="Resolution", loc="upper left")

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
