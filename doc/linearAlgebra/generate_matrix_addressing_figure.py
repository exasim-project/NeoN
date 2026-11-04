# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Generates three figures illustrating NeoN matrix addressing
for the 8-cell 2-rank 1D example documented in matrixAddressing.rst.

Figures produced:
  matrix_cell_connectivity.png   — mesh topology and face labelling
  matrix_csr_pattern.png         — 4×4 CSR non-zero pattern (rank-0 local matrix)
  matrix_offset_table.png        — offset arrays summary table

Run with:
    python generate_matrix_addressing_figure.py

Requires: matplotlib, numpy
"""

import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

DIR = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# 8-cell 2-rank topology
# ---------------------------------------------------------------------------
# Rank 0: C0, C1, C2, C3   (4 cells)
# Rank 1: C4, C5, C6, C7   (4 cells — shown as ghost on rank 0's figure)
#
# Internal faces on rank 0:
#   f0: own=C0, nei=C1
#   f1: own=C1, nei=C2
#   f2: own=C2, nei=C3
#
# Processor boundary face (rank 0 ↔ rank 1):
#   pf0: own=C3 (rank 0), nei=C4 (rank 1)
#
# Physical boundary face on rank 0:
#   bf0: left wall of C0

# CSR for rank-0 local 4×4 matrix:
#   Row C0: [A[C0,C0], A[C0,C1]]                          flat 0,1
#   Row C1: [A[C1,C0], A[C1,C1], A[C1,C2]]               flat 2,3,4
#   Row C2: [A[C2,C1], A[C2,C2], A[C2,C3]]               flat 5,6,7
#   Row C3: [A[C3,C2], A[C3,C3]]                          flat 8,9
#
ROW_OFFS = [0, 2, 5, 8, 10]
COL_IDXS = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3]

# (row, col) of each non-zero in the 4×4 matrix
NNZ_POS = [
    (0, 0), (0, 1),
    (1, 0), (1, 1), (1, 2),
    (2, 1), (2, 2), (2, 3),
    (3, 2), (3, 3),
]
NNZ_TYPE = ["d", "u", "l", "d", "u", "l", "d", "u", "l", "d"]
NNZ_LABEL = [
    "A[C0,C0]", "A[C0,C1]",
    "A[C1,C0]", "A[C1,C1]", "A[C1,C2]",
    "A[C2,C1]", "A[C2,C2]", "A[C2,C3]",
    "A[C3,C2]", "A[C3,C3]",
]

# Offset arrays for rank-0 matrix:
DIAG_OFFSET = [0, 1, 1, 1]   # diagOffset[Ci]
OWNER_OFFSET = [1, 2, 2]      # ownerOffset[f0,f1,f2] → upper triangular A[own,nei]
NEIGH_OFFSET = [0, 0, 0]      # neighbourOffset[f0,f1,f2] → lower triangular A[nei,own]

COLOR = {"d": "#FFA500", "u": "#ADD8E6", "l": "#90EE90", "zero": "#EEEEEE"}

# ---------------------------------------------------------------------------
# Figure 1 — Cell connectivity
# ---------------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(14, 5))
fig1.suptitle(
    "NeoN Matrix Addressing — 8-Cell 2-Rank Mesh Topology",
    fontsize=13, fontweight="bold",
)
ax.set_xlim(-1.8, 12.5)
ax.set_ylim(-1.5, 2.8)
ax.set_aspect("equal")
ax.set_axis_off()

cell_w, cell_h = 1.2, 1.0
gap = 0.4   # gap between cells
rank0_x = [i * (cell_w + gap) for i in range(4)]   # C0..C3
rank1_x = [rank0_x[-1] + cell_w + gap * 3 + i * (cell_w + gap) for i in range(4)]  # C4..C7
cell_y = 0.5

rank0_color = "#C8E6FA"
rank1_color = "#FFE4E4"

# Draw rank-0 cells
for i, cx in enumerate(rank0_x):
    rect = FancyBboxPatch(
        (cx, cell_y), cell_w, cell_h,
        boxstyle="round,pad=0.05", linewidth=1.8,
        edgecolor="steelblue", facecolor=rank0_color,
    )
    ax.add_patch(rect)
    ax.text(cx + cell_w / 2, cell_y + cell_h / 2, f"C{i}",
            ha="center", va="center", fontsize=12, fontweight="bold")

# Rank-0 label banner
bx0 = rank0_x[0] - 0.15
bw0 = rank0_x[-1] + cell_w + 0.15 - bx0
ax.add_patch(FancyBboxPatch((bx0, cell_y + cell_h + 0.12), bw0, 0.32,
                             boxstyle="round,pad=0.04", linewidth=1,
                             edgecolor="steelblue", facecolor="#EAF4FB"))
ax.text(bx0 + bw0 / 2, cell_y + cell_h + 0.28, "Rank 0",
        ha="center", va="center", fontsize=10, fontweight="bold", color="steelblue")

# Draw rank-1 cells (ghost/remote — dashed)
for i, cx in enumerate(rank1_x):
    rect = FancyBboxPatch(
        (cx, cell_y), cell_w, cell_h,
        boxstyle="round,pad=0.05", linewidth=1.8,
        edgecolor="crimson", facecolor=rank1_color, linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(cx + cell_w / 2, cell_y + cell_h / 2, f"C{4+i}",
            ha="center", va="center", fontsize=12, fontweight="bold", color="crimson")

# Rank-1 label banner
bx1 = rank1_x[0] - 0.15
bw1 = rank1_x[-1] + cell_w + 0.15 - bx1
ax.add_patch(FancyBboxPatch((bx1, cell_y + cell_h + 0.12), bw1, 0.32,
                             boxstyle="round,pad=0.04", linewidth=1,
                             edgecolor="crimson", facecolor="#FFF0F0"))
ax.text(bx1 + bw1 / 2, cell_y + cell_h + 0.28, "Rank 1",
        ha="center", va="center", fontsize=10, fontweight="bold", color="crimson")

# Internal faces on rank 0 (f0, f1, f2)
for fi in range(3):
    fx = rank0_x[fi] + cell_w + gap / 2
    ax.axvline(fx, ymin=0.37, ymax=0.62, color="steelblue", linewidth=2, linestyle="--")
    ax.text(fx, cell_y + cell_h + 0.55, f"f{fi}",
            ha="center", va="bottom", fontsize=9.5, color="steelblue", fontweight="bold")
    ax.text(fx, cell_y - 0.25,
            f"own=C{fi}\nnei=C{fi+1}",
            ha="center", va="top", fontsize=7.5, color="dimgray")

# Internal faces on rank 1 (f3, f4, f5 — shown for completeness)
for fi in range(3):
    fx = rank1_x[fi] + cell_w + gap / 2
    ax.axvline(fx, ymin=0.37, ymax=0.62, color="crimson", linewidth=1.5, linestyle=":")
    ax.text(fx, cell_y + cell_h + 0.55, f"f{fi+3}",
            ha="center", va="bottom", fontsize=9.5, color="crimson", fontweight="bold")
    ax.text(fx, cell_y - 0.25,
            f"own=C{4+fi}\nnei=C{4+fi+1}",
            ha="center", va="top", fontsize=7.5, color="#AA6666")

# Processor boundary face (pf0)
pfx = rank0_x[-1] + cell_w + gap * 1.5
ax.axvline(pfx, ymin=0.30, ymax=0.70, color="darkorange", linewidth=3, linestyle="-.")
ax.text(pfx, cell_y + cell_h + 0.55, "pf0",
        ha="center", va="bottom", fontsize=9.5, color="darkorange", fontweight="bold")
ax.text(pfx, cell_y - 0.25,
        "PROC boundary\nown=C3 | nei=C4",
        ha="center", va="top", fontsize=7.5, color="darkorange")

# Physical boundary face (bf0, left wall of C0)
wfx = rank0_x[0] - 0.45
for dy in np.linspace(cell_y + 0.05, cell_y + cell_h - 0.05, 7):
    ax.plot([wfx, wfx + 0.3], [dy, dy + 0.12], color="forestgreen", linewidth=1.2)
ax.axvline(wfx + 0.3, ymin=0.37, ymax=0.62, color="forestgreen", linewidth=2.5)
ax.text(wfx - 0.05, cell_y + cell_h / 2, "bf0\n(BC_wall)",
        ha="right", va="center", fontsize=9, color="forestgreen", fontweight="bold")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=rank0_color, edgecolor="steelblue", label="Rank-0 cells (C0–C3)"),
    mpatches.Patch(facecolor=rank1_color, edgecolor="crimson", label="Rank-1 cells (C4–C7, ghost)"),
    mpatches.Patch(facecolor="steelblue", alpha=0.7, label="Internal face (rank 0)"),
    mpatches.Patch(facecolor="darkorange", alpha=0.7, label="Processor boundary (pf0)"),
    mpatches.Patch(facecolor="forestgreen", alpha=0.7, label="Physical BC (bf0)"),
]
ax.legend(handles=legend_handles, loc="lower center", fontsize=8.5,
          framealpha=0.9, bbox_to_anchor=(0.5, -0.12), ncol=3)

plt.tight_layout()
fig1.savefig(DIR / "matrix_cell_connectivity.png", dpi=150, bbox_inches="tight")
print(f"Saved matrix_cell_connectivity.png")

# ---------------------------------------------------------------------------
# Figure 2 — CSR non-zero pattern (rank-0 local 4×4 matrix)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.suptitle(
    "CSR Non-zero Pattern — Rank-0 Local Matrix (4×4)\n"
    "(numbers = flat values[] index)",
    fontsize=12, fontweight="bold",
)

nrows, ncols = 4, 4
cell_size = 1.0
padding = 0.06

ax2.set_xlim(-0.5, ncols * cell_size + 1.6)
ax2.set_ylim(-0.7, nrows * cell_size + 1.0)
ax2.set_aspect("equal")
ax2.set_axis_off()

for row in range(nrows):
    for col in range(ncols):
        x = col * cell_size
        y = (nrows - 1 - row) * cell_size

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
            boxstyle="round,pad=0.04", linewidth=1.2,
            edgecolor="gray", facecolor=fc,
        )
        ax2.add_patch(rect)

        if entry_idx is not None:
            ax2.text(x + cell_size / 2, y + cell_size * 0.62,
                     str(entry_idx),
                     ha="center", va="center", fontsize=14, fontweight="bold")
            ax2.text(x + cell_size / 2, y + cell_size * 0.25,
                     NNZ_LABEL[entry_idx],
                     ha="center", va="center", fontsize=6.5, color="dimgray")

# Row labels + rowOffs ranges
row_labels = ["C0", "C1", "C2", "C3"]
for row, label in enumerate(row_labels):
    y = (nrows - 1 - row) * cell_size + cell_size / 2
    ax2.text(-0.38, y, label, ha="right", va="center", fontsize=11, fontweight="bold")
    start = ROW_OFFS[row]
    end = ROW_OFFS[row + 1]
    ax2.text(ncols * cell_size + 0.12, y,
             f"[{start}..{end-1}]",
             ha="left", va="center", fontsize=8.5, color="dimgray")

ax2.text(ncols * cell_size + 0.12, nrows * cell_size + 0.35,
         "rowOffs\nrange", ha="left", va="center", fontsize=8, color="dimgray")

# Column headers
col_labels = ["col 0\n(C0)", "col 1\n(C1)", "col 2\n(C2)", "col 3\n(C3)"]
for col, label in enumerate(col_labels):
    x = col * cell_size + cell_size / 2
    ax2.text(x, nrows * cell_size + 0.12, label,
             ha="center", va="bottom", fontsize=9)

# Annotation: ownerOffset points into upper tri, neighbourOffset into lower tri
ax2.annotate(
    "ownerOffset[f] → upper tri A[own,nei]\n(own < nei by construction)",
    xy=(2.5 * cell_size, 3 * cell_size + 0.5),
    xytext=(3.4 * cell_size + 0.1, 3 * cell_size + 0.75),
    fontsize=7.5, color="steelblue",
    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
    ha="left", va="center",
)
ax2.annotate(
    "neighbourOffset[f] → lower tri A[nei,own]\n(nei > own by construction)",
    xy=(0.5 * cell_size, 2 * cell_size + 0.5),
    xytext=(3.4 * cell_size + 0.1, 2 * cell_size + 0.25),
    fontsize=7.5, color="forestgreen",
    arrowprops=dict(arrowstyle="->", color="forestgreen", lw=1.2),
    ha="left", va="center",
)

legend_handles2 = [
    mpatches.Patch(facecolor=COLOR["d"], edgecolor="gray", label="Diagonal"),
    mpatches.Patch(facecolor=COLOR["u"], edgecolor="gray", label="Upper tri A[own,nei]  (ownerOffset)"),
    mpatches.Patch(facecolor=COLOR["l"], edgecolor="gray", label="Lower tri A[nei,own]  (neighbourOffset)"),
    mpatches.Patch(facecolor=COLOR["zero"], edgecolor="gray", label="Zero (structural)"),
]
ax2.legend(handles=legend_handles2, loc="lower center", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.35, -0.12), ncol=2)

plt.tight_layout()
fig2.savefig(DIR / "matrix_csr_pattern.png", dpi=150, bbox_inches="tight")
print(f"Saved matrix_csr_pattern.png")

# ---------------------------------------------------------------------------
# Figure 3 — Offset arrays summary table
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.suptitle(
    "Offset Arrays — Rank-0 8-Cell Example",
    fontsize=12, fontweight="bold",
)
ax3.set_axis_off()

table_data = [
    ["Array", "Values", "Notes"],
    ["rowOffs", "[0, 2, 5, 8, 10]", "size = nCells+1 = 5"],
    ["colIdxs", "[0,1, 0,1,2, 1,2,3, 2,3]", "size = nnz = 10"],
    ["diagOffset", "[0, 1, 1, 1]", "one entry per cell (C0..C3)"],
    ["ownerOffset", "[1, 2, 2]", "f0,f1,f2 → A[own,nei]  (upper tri)"],
    ["neighbourOffset", "[0, 0, 0]", "f0,f1,f2 → A[nei,own]  (lower tri)"],
    ["", "", ""],
    ["bRowIdx", "[0]", "bf0 owner = C0"],
    ["bColIdx", "[0]", "= C0 + diagOffset[C0] = 0+0"],
    ["", "", ""],
    ["pRowIdx", "[3]", "pf0 local owner = C3"],
    ["pColIdx", "[4]", "= C3 + diagOffset[C3] = 3+1"],
]

col_widths = [0.30, 0.40, 0.30]
row_height = 0.075
x_start = 0.02
y_start = 0.93

header_color = "#D0D8E8"
alt_color = "#F5F5F5"

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
        ax3.text(
            x + cw * 0.04, y - row_height * 0.38,
            cell_text,
            transform=ax3.transAxes,
            ha="left", va="center",
            fontsize=fs, fontweight=fw,
            fontfamily="monospace" if ci <= 1 else "sans-serif",
        )
        x += cw

# Separator lines
sep_y = y_start - 6 * row_height
ax3.axhline(sep_y, xmin=x_start, xmax=x_start + sum(col_widths),
            color="steelblue", linewidth=0.8)
sep_y2 = y_start - 9 * row_height
ax3.axhline(sep_y2, xmin=x_start, xmax=x_start + sum(col_widths),
            color="darkorange", linewidth=0.8)

note_y = y_start - (len(table_data) + 0.5) * row_height
ax3.text(
    x_start, note_y,
    "COO colIdx = celli + diagOffset[celli]\n"
    "(≠ rowOffs[celli] + diagOffset[celli] for celli > 0;\n"
    " see note in matrixAddressing.rst)",
    transform=ax3.transAxes,
    ha="left", va="top", fontsize=7.5,
    color="saddlebrown", style="italic",
)

legend_handles3 = [
    mpatches.Patch(facecolor="steelblue", alpha=0.7, label="Physical BC (bRowIdx/bColIdx)"),
    mpatches.Patch(facecolor="darkorange", alpha=0.7, label="Proc. BC (pRowIdx/pColIdx)"),
]
ax3.legend(handles=legend_handles3, loc="lower left", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.0, -0.06))

plt.tight_layout()
fig3.savefig(DIR / "matrix_offset_table.png", dpi=150, bbox_inches="tight")
print(f"Saved matrix_offset_table.png")
