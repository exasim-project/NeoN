# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Create a uniform 3D hex mesh and write to VTU, CGNS, and VTK HDF."""

import neon
import pyvista as pv

exec = neon.SerialExecutor()
mesh = neon.create_uniform_3d_mesh(exec, nx=4, ny=4, nz=4)

print(f"Cells: {mesh.n_cells()}, Faces: {mesh.n_faces()}")
print(f"Boundaries: {mesh.n_boundaries()}, Boundary faces: {mesh.n_boundary_faces()}")

neon.write_vtu(mesh, "grid3d.vtu")
neon.write_cgns(mesh, "grid3d.cgns")
neon.write_vtk_hdf(mesh, "grid3d.vtkhdf")

print("Written: grid3d.vtu, grid3d.cgns, grid3d.vtkhdf")

grid = pv.read("grid3d.vtu")
grid.plot(show_edges=True, color="lightblue")
