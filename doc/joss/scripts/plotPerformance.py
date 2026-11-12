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

Top:    assembly time per cell over the number of cells for every GPU file and
        for the CPUExecutor of every CPU using 1 and --mt-threads threads.
Bottom: CPUExecutor speedup relative to a single thread, one panel per CPU.

Usage:
    cd doc/joss/figures
    python3 ../scripts/plotPerformance.py \\
        --gpu A100=data/momentum_nvidia_a100.1.csv --gpu H100=data/momentum_nvidia_h100.1.csv \\
        --gpu H200=data/momentum_nvidia_h200.1.csv --gpu MI210=data/momentum_amd_mi210.1.csv \\
        --gpu PVC=data/momentum_intel_pvc.1.csv \\
        --cpu "EPYC 9555=data/momentum_nvidia_h200" \\
        --cpu "Xeon Gold 6438Y+=data/momentum_intel_xeon6438y" \\
        --cpu "Xeon Platinum 8480+=data/momentum_intel_xeon8480" \\
        --output neon_performance.png

Every --cpu takes LABEL=PREFIX and reads all CPU files named <PREFIX>.<threads>.csv.
"""

import argparse
import glob
import re
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

TEXT, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GPU_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#8c7a00", "#56b4e9", "#e69f00"]
CPU_COLORS = ["#222222", "#8c4f12", "#4d7c2a", "#7c3aed", "#b45309"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LINE = dict(markersize=2.5, linewidth=1.1)


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", action="append", default=[], metavar="LABEL=CSV")
    parser.add_argument("--cpu", action="append", required=True, metavar="LABEL=PREFIX")
    parser.add_argument(
        "--mt-threads", type=int, default=16, help="threads of the multi-core curves"
    )
    parser.add_argument("--min-res", type=int, default=75)
    parser.add_argument("--max-res", type=int, default=275)
    parser.add_argument("--section", default="without RHS")
    parser.add_argument("--output", type=Path, default=Path("neon_performance.png"))
    args = parser.parse_args()

    def in_range(df):
        return df[df["Resolution"].between(args.min_res, args.max_res)]

    cpu = []
    for spec in args.cpu:
        name, prefix = spec.split("=", 1)
        paths = sorted(glob.glob(f"{prefix}.*.csv"))
        if not paths:
            raise SystemExit(f"no files match {prefix}.<threads>.csv")
        for path in paths:
            df = load(path, args.section)
            df = df[df["benchmark_name"] == "CPUExecutor"]
            df["Threads"] = threads_of(path)
            cpu.append(df.assign(CPU=name))
    cpu = in_range(pd.concat(cpu, ignore_index=True))

    single, multi = "1 thread", f"{args.mt_threads} threads"
    dashes = {"GPU": "", single: (1, 1.5), multi: (4, 2)}
    markers = {"GPU": "o", single: "s", multi: "o"}

    curves = []
    for spec in args.gpu:
        label, path = spec.split("=", 1)
        df = load(path, args.section)
        df = df[df["benchmark_name"] == "GPUExecutor"]
        curves.append(df.assign(Label=label, Style="GPU"))
    for name, df in cpu.groupby("CPU", sort=False):
        curves.append(df[df["Threads"] == 1].assign(Label=name, Style=single))
        curves.append(df[df["Threads"] == args.mt_threads].assign(Label=name, Style=multi))
    curves = in_range(pd.concat(curves))

    baseline = cpu[cpu["Threads"] == 1].set_index(["CPU", "Resolution"])["avg_runtime"]
    baseline = cpu.join(baseline.rename("Baseline"), on=["CPU", "Resolution"])["Baseline"]
    cpu["Speedup [-]"] = baseline / cpu["avg_runtime"]

    plt.rcParams.update({
        "font.size": 7.5, "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT_SECONDARY, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "axes.linewidth": 0.6, "axes.labelpad": 2,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
    })
    cpus = list(cpu["CPU"].unique())
    fig = plt.figure(figsize=(6.5, 6.6))
    grid = fig.add_gridspec(2, len(cpus), height_ratios=(1.7, 1))
    top = fig.add_subplot(grid[0, :])
    bottoms = [fig.add_subplot(grid[1, 0])]
    bottoms += [fig.add_subplot(grid[1, i], sharey=bottoms[0]) for i in range(1, len(cpus))]

    sns.lineplot(
        data=curves, x="Cells", y="Time/Cell", hue="Label", style="Style",
        markers=markers, dashes=dashes, legend=False, ax=top,
        palette=GPU_COLORS[: len(args.gpu)] + CPU_COLORS[: len(cpus)], **LINE,
    )
    for line in top.get_lines():
        line.set(markeredgecolor=line.get_color(), markeredgewidth=1.0)
        if line.get_linestyle() != "-":
            line.set(markerfacecolor="white", markeredgewidth=0.8)
    top.set(xscale="log", yscale="log", xlabel="Number of cells")
    top.set_ylabel("Assembly time per cell [ns]")
    top.yaxis.set_major_formatter(ScalarFormatter())
    top.set_ylim(top=curves["Time/Cell"].max() * 3)
    top.set_box_aspect(0.9)
    top.set_anchor((0.3, 0.5))
    top.spines[["top", "right"]].set_visible(True)
    top.tick_params(which="both", top=True, right=True)
    top.text(0.01, 0.95, "(a)", transform=top.transAxes, va="top")

    def heading(text):
        return Line2D([], [], linestyle="none", label=text)

    handles = [heading("GPU")]
    handles += [
        Line2D([], [], color=color, marker="o", label=spec.split("=")[0], **LINE)
        for spec, color in zip(args.gpu, GPU_COLORS)
    ]
    handles += [heading(""), heading("CPU")]
    handles += [
        Line2D([], [], color=color, marker="s", markersize=1.75 * LINE["markersize"],
               linestyle="none", label=name)
        for name, color in zip(cpus, CPU_COLORS)
    ]
    handles += [heading(""), heading("CPUExecutor")]
    handles += [
        Line2D([], [], color=CPU_COLORS[0], linestyle=(0, dashes[style]), marker=markers[style],
               markerfacecolor="white", markeredgewidth=0.8, label=style, **LINE)
        for style in (single, multi)
    ]
    legend = top.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5))
    legend.set_in_layout(False)

    n_res = cpu["Resolution"].nunique()
    palette = [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(n_res - 1, 1))] for i in range(n_res)]
    for i, (bottom, name) in enumerate(zip(bottoms, cpus)):
        sns.barplot(
            data=cpu[cpu["CPU"] == name], x="Threads", y="Speedup [-]", hue="Resolution",
            palette=palette, width=0.85, errorbar=None, saturation=1, edgecolor="white",
            linewidth=0.3, legend=(i == 0), ax=bottom,
        )
        bottom.set(xlabel="Number of threads" if i == len(cpus) // 2 else "", ylabel="")
        bottom.set_title(name, fontsize=8.5, pad=8.5)
        bottom.text(0.03, 0.95, f"({ascii_lowercase[i + 1]})", transform=bottom.transAxes, va="top")
        if i > 0:
            bottom.tick_params(labelleft=False)
    bottoms[0].set_ylabel("Speedup relative to 1 thread")
    handles, labels = bottoms[0].get_legend_handles_labels()
    bottoms[0].get_legend().remove()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0),
        title="Cube resolution N (number of cells = N³)",
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.5)
    fig.savefig(args.output, dpi=300)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
