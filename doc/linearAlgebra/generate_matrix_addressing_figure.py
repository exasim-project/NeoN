# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Generates matrix_addressing_figure.png illustrating NeoN matrix addressing
for the 3-cell 1D example documented in matrixAddressing.rst.

Run with:
    python generate_matrix_addressing_figure.py

Requires: matplotlib, numpy
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUTPUT = pathlib.Path(__file__).parent / "matrix_addressing_figure.png"

# ---------------------------------------------------------------------------
# Data for the 3-cell example
# ---------------------------------------------------------------------------
ROW_OFFS = [0, 2, 5, 7]
COL_IDXS = [0, 1, 0, 1, 2, 1, 2]

# (row, col) of each non-zero in the 3x3 matrix
NNZ_POS = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
# type: 'd'=diagonal, 'u'=upper triangular, 'l'=lower triangular
NNZ_TYPE = ["d", "u", "l", "d", "u", "l", "d"]
NNZ_LABEL = [
    "A[C0,C0]", "A[C0,C1]",
    "A[C1,C0]", "A[C1,C1]", "A[C1,C2]",
    "A[C2,C1]", "A[C2,C2]",
]

COLOR = {"d": "#FFA500", "u": "#ADD8E6", "l": "#90EE90", "zero": "#EEEEEE"}

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 6))
fig.suptitle(
    "NeoN Matrix Addressing — 3-Cell 1D Example",
    fontsize=14, fontweight="bold", y=1.01,
)

# ---------------------------------------------------------------------------
# Panel 1 — Mesh topology
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlim(-1.5, 4.5)
ax1.set_ylim(-1.2, 2.2)
ax1.set_aspect("equal")
ax1.set_axis_off()
ax1.set_title("Mesh Topology", fontsize=12, fontweight="bold")

cell_w, cell_h = 1.0, 1.0
cell_x = [0.0, 1.5, 3.0]
cell_y = 0.5

cell_color = "#C8E6FA"
for i, cx in enumerate(cell_x):
    rect = FancyBboxPatch(
        (cx, cell_y), cell_w, cell_h,
        boxstyle="round,pad=0.05", linewidth=1.5,
        edgecolor="steelblue", facecolor=cell_color,
    )
    ax1.add_patch(rect)
    ax1.text(
        cx + cell_w / 2, cell_y + cell_h / 2, f"C{i}",
        ha="center", va="center", fontsize=13, fontweight="bold",
    )

# Internal faces
for fi, (x1, x2) in enumerate(zip(cell_x[:-1], cell_x[1:])):
    fx = (x1 + cell_w + x2) / 2
    ax1.axvline(fx, ymin=0.35, ymax=0.72, color="steelblue", linewidth=2, linestyle="--")
    ax1.text(fx, cell_y + cell_h + 0.18, f"f{fi}", ha="center", va="bottom",
             fontsize=10, color="steelblue", fontweight="bold")
    # own/nei label
    own, nei = fi, fi + 1
    ax1.text(fx, cell_y - 0.25,
             f"own=C{own}\nnei=C{nei}",
             ha="center", va="top", fontsize=7.5, color="dimgray")

# Physical boundary (left of C0)
bx = cell_x[0] - 0.55
for dy in np.linspace(cell_y + 0.05, cell_y + cell_h - 0.05, 7):
    ax1.plot([bx, bx + 0.3], [dy, dy + 0.12], color="forestgreen", linewidth=1.2)
ax1.axvline(bx + 0.3, ymin=0.35, ymax=0.72, color="forestgreen", linewidth=2.5)
ax1.text(bx - 0.05, cell_y + cell_h / 2, "BC_wall\n(bf0)",
         ha="right", va="center", fontsize=9, color="forestgreen", fontweight="bold")

# Processor boundary (right of C2)
px = cell_x[-1] + cell_w + 0.25
ghost_rect = FancyBboxPatch(
    (px, cell_y), cell_w * 0.65, cell_h,
    boxstyle="round,pad=0.05", linewidth=1.5,
    edgecolor="crimson", facecolor="#FFE4E4", linestyle="--",
)
ax1.add_patch(ghost_rect)
ax1.text(px + cell_w * 0.33, cell_y + cell_h / 2,
         "rank\n1", ha="center", va="center",
         fontsize=8, color="crimson")
ax1.axvline(px, ymin=0.35, ymax=0.72, color="crimson", linewidth=2.5, linestyle="--")
ax1.text(px + cell_w * 0.33, cell_y + cell_h + 0.18, "pf0",
         ha="center", va="bottom", fontsize=10, color="crimson", fontweight="bold")
ax1.text(px + cell_w * 0.33, cell_y - 0.25,
         "PROC boundary\nown=C2",
         ha="center", va="top", fontsize=7.5, color="crimson")

# Legend
legend_handles = [
    mpatches.Patch(facecolor="forestgreen", edgecolor="forestgreen", label="Physical BC (bf0)"),
    mpatches.Patch(facecolor="crimson", edgecolor="crimson", label="Proc. BC (pf0)"),
    mpatches.Patch(facecolor="steelblue", edgecolor="steelblue", label="Internal face"),
]
ax1.legend(handles=legend_handles, loc="lower center", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

# ---------------------------------------------------------------------------
# Panel 2 — CSR non-zero pattern
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(1, 3, 2)
nrows, ncols = 3, 3
cell_size = 0.9
padding = 0.05

ax2.set_xlim(-0.3, ncols * cell_size + 1.2)
ax2.set_ylim(-0.6, nrows * cell_size + 0.9)
ax2.set_aspect("equal")
ax2.set_axis_off()
ax2.set_title("CSR Non-zero Pattern\n(numbers = flat values index)", fontsize=11, fontweight="bold")

# Draw the 3x3 grid
for row in range(nrows):
    for col in range(ncols):
        x = col * cell_size
        y = (nrows - 1 - row) * cell_size

        # Determine if this (row,col) is a non-zero
        entry_idx = None
        entry_type = "zero"
        for k, (r, c) in enumerate(NNZ_POS):
            if r == row and c == col:
                entry_idx = k
                entry_type = NNZ_TYPE[k]
                break

        fc = COLOR[entry_type]
        rect = FancyBboxPatch(
            (x + padding, y + padding),
            cell_size - 2 * padding, cell_size - 2 * padding,
            boxstyle="round,pad=0.03", linewidth=1.2,
            edgecolor="gray", facecolor=fc,
        )
        ax2.add_patch(rect)

        if entry_idx is not None:
            # Flat value index
            flat_idx = entry_idx
            ax2.text(x + cell_size / 2, y + cell_size * 0.62,
                     str(flat_idx),
                     ha="center", va="center", fontsize=13, fontweight="bold")
            ax2.text(x + cell_size / 2, y + cell_size * 0.28,
                     NNZ_LABEL[entry_idx],
                     ha="center", va="center", fontsize=6.5, color="dimgray")

# Row labels (cell names)
row_labels = ["C0", "C1", "C2"]
for row, label in enumerate(row_labels):
    y = (nrows - 1 - row) * cell_size + cell_size / 2
    ax2.text(-0.22, y, label, ha="right", va="center", fontsize=11, fontweight="bold")

# Row offset annotations on the right
for row in range(nrows):
    y = (nrows - 1 - row) * cell_size + cell_size / 2
    start = ROW_OFFS[row]
    end = ROW_OFFS[row + 1]
    ax2.text(ncols * cell_size + 0.12, y,
             f"[{start}..{end-1}]",
             ha="left", va="center", fontsize=8, color="dimgray")
ax2.text(ncols * cell_size + 0.12, nrows * cell_size + 0.3,
         "rowOffs\nrange", ha="left", va="center", fontsize=7.5, color="dimgray")

# Column labels (matrix entry headers)
col_labels = ["col 0\n(C0)", "col 1\n(C1)", "col 2\n(C2)"]
for col, label in enumerate(col_labels):
    x = col * cell_size + cell_size / 2
    ax2.text(x, nrows * cell_size + 0.1, label,
             ha="center", va="bottom", fontsize=9)

# Legend
legend_handles2 = [
    mpatches.Patch(facecolor=COLOR["d"], edgecolor="gray", label="Diagonal"),
    mpatches.Patch(facecolor=COLOR["u"], edgecolor="gray", label="Upper triangular A[own,nei]"),
    mpatches.Patch(facecolor=COLOR["l"], edgecolor="gray", label="Lower triangular A[nei,own]"),
    mpatches.Patch(facecolor=COLOR["zero"], edgecolor="gray", label="Zero"),
]
ax2.legend(handles=legend_handles2, loc="lower center", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.4, -0.12), ncol=2)

# ---------------------------------------------------------------------------
# Panel 3 — Offset array summary table
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_axis_off()
ax3.set_title("Offset Arrays", fontsize=12, fontweight="bold")

table_data = [
    ["Array", "Values", "Notes"],
    ["rowOffs", "[0, 2, 5, 7]", "size = nCells+1"],
    ["colIdxs", "[0,1, 0,1,2, 1,2]", "size = nnz = 7"],
    ["diagOffset", "[0, 1, 1]", "per cell"],
    ["ownerOffset", "[1, 2]", "f0,f1 → A[own,nei]"],
    ["neighbourOffset", "[0, 0]", "f0,f1 → A[nei,own]"],
    ["", "", ""],
    ["bRowIdx", "[0]", "bf0 owner = C0"],
    ["bColIdx", "[0]", "= C0+diagOff[C0]"],
    ["", "", ""],
    ["pRowIdx", "[2]", "pf0 owner = C2"],
    ["pColIdx", "[3]", "= C2+diagOff[C2]"],
]

col_widths = [0.36, 0.36, 0.28]
row_height = 0.072
x_start = 0.02
y_start = 0.93

header_color = "#D0D8E8"
alt_color = "#F5F5F5"
sep_color = "#E8E8E8"

for ri, row in enumerate(table_data):
    y = y_start - ri * row_height
    is_header = ri == 0
    is_sep = all(v == "" for v in row)

    if is_sep:
        continue

    bg = header_color if is_header else (alt_color if ri % 2 == 0 else "white")
    total_w = sum(col_widths)
    bg_rect = FancyBboxPatch(
        (x_start, y - row_height * 0.85), total_w, row_height * 0.9,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=bg, transform=ax3.transAxes, clip_on=False,
    )
    ax3.add_patch(bg_rect)

    x = x_start
    for ci, (cell_text, cw) in enumerate(zip(row, col_widths)):
        fw = "bold" if is_header or ci == 0 else "normal"
        fs = 8.5 if ci == 0 else 8
        color = "black"
        ax3.text(
            x + cw * 0.05, y - row_height * 0.35,
            cell_text,
            transform=ax3.transAxes,
            ha="left", va="center",
            fontsize=fs, fontweight=fw, color=color,
            fontfamily="monospace" if ci <= 1 else "sans-serif",
        )
        x += cw

# Separator line above bRowIdx
sep_y = y_start - 6 * row_height
ax3.axhline(sep_y, xmin=x_start, xmax=x_start + sum(col_widths),
            color="steelblue", linewidth=0.8)

sep_y2 = y_start - 9 * row_height
ax3.axhline(sep_y2, xmin=x_start, xmax=x_start + sum(col_widths),
            color="crimson", linewidth=0.8)

# Formula note
note_y = y_start - (len(table_data) + 0.5) * row_height
ax3.text(
    x_start, note_y,
    "COO colIdx = celli + diagOffset[celli]\n"
    "(≠ rowOffs[celli] + diagOffset[celli] for celli > 0;\n"
    " see note in matrixAddressing.rst)",
    transform=ax3.transAxes,
    ha="left", va="top", fontsize=7.5,
    color="saddlebrown",
    style="italic",
    wrap=True,
)

# Color legend for boundary types
legend_handles3 = [
    mpatches.Patch(facecolor="forestgreen", alpha=0.7, label="Physical BC (bRowIdx/bColIdx)"),
    mpatches.Patch(facecolor="crimson", alpha=0.7, label="Proc. BC (pRowIdx/pColIdx)"),
]
ax3.legend(handles=legend_handles3, loc="lower left", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.0, -0.04))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT}")
