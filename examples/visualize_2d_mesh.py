# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Create a uniform 2D mesh (OpenFOAM-style hex slab) and write to VTM, CGNS, and VTK HDF."""

from pathlib import Path

import neon
import pyvista as pv

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

exec = neon.SerialExecutor()
mesh = neon.create_uniform_2d_mesh(exec, nx=8, ny=8)

print(f"Cells: {mesh.n_cells()}, Faces: {mesh.n_faces()}")
print(f"Boundaries: {mesh.n_boundaries()}, Boundary faces: {mesh.n_boundary_faces()}")

neon.write_vtm(mesh, str(output_dir / "grid2d.vtm"))
neon.write_cgns(mesh, str(output_dir / "grid2d.cgns"))
neon.write_vtk_hdf(mesh, str(output_dir / "grid2d.vtkhdf"))

print(f"Written: {output_dir}/grid2d.vtm, {output_dir}/grid2d.cgns, {output_dir}/grid2d.vtkhdf")

grid = pv.read(str(output_dir / "grid2d.vtm"))
grid.plot(show_edges=True, color="lightblue")
