# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Partition a uniform 2D mesh into N parts, write each sub-mesh to VTK HDF, and visualize."""

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
    neon.write_vtm(sub, vtm_path)
    print(f"  Part {part_id}: {sub.n_cells()} cells → partition_{part_id}.vtkhdf")
    grid = pv.read(vtm_path)
    plotter.add_mesh(grid, show_edges=True, color=colours[part_id % len(colours)], label=f"Part {part_id}")

plotter.add_legend()
plotter.show()
