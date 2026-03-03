# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Partition a uniform 2D mesh into N parts, write each sub-mesh to VTK HDF, and visualize."""

import numpy as np
import neon
import pyvista as pv

exec_ = neon.SerialExecutor()
mesh = neon.create_uniform_2d_mesh(exec_, nx=8, ny=8)
print(f"Global mesh: {mesh.n_cells()} cells, {mesh.n_faces()} faces")

n_parts = 4
cell_part = neon.partition_mesh(mesh, n_parts)
print(f"Partitioned into {n_parts} parts")

colours = ["tomato", "steelblue", "mediumseagreen", "goldenrod"]
plotter = pv.Plotter()
for part_id in range(n_parts):
    sub = neon.extract_sub_mesh(mesh, cell_part, part_id)
    neon.write_vtk_hdf(sub, f"partition_{part_id}.vtkhdf")
    vtm_path = f"partition_{part_id}.vtm"
    neon.write_vtm(sub, vtm_path, include_ghosts=True)

    # Get per-neighbor proc patch names and global cell IDs
    patch_names = neon.get_patch_names(sub)
    global_ids = neon.get_global_cell_ids(sub)
    proc_patches = [n for n in patch_names if n.startswith("proc")]
    print(
        f"  Part {part_id}: {sub.n_cells()} cells, "
        f"proc patches: {proc_patches}, "
        f"global ids: {global_ids}"
    )

    if part_id == 0:
        grid = pv.read(vtm_path)
        internal = grid[0]
        ghost_ids = neon.get_ghost_cell_ids(sub)
        all_ids = list(global_ids) + list(ghost_ids)
        internal.cell_data["cell_id"] = np.array(all_ids)
        plotter.add_mesh(
            internal,
            scalars="ghostCells",
            show_edges=True,
            label=f"Part {part_id}",
        )

plotter.add_legend()
plotter.show()
