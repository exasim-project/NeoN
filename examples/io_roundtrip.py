# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Read a CGNS mesh, write it to VTU, and visualize with pyvista."""

import os

import neon
import pyvista as pv

# Locate the cavity mesh relative to the repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..", "..")
cgns_path = os.path.join(repo_root, "test", "mesh", "unstructured", "io", "meshFiles", "cavity2D.cgns")

exec = neon.SerialExecutor()
mesh = neon.read_cgns(cgns_path, exec)

print(f"Cells: {mesh.n_cells}, Faces: {mesh.n_faces}")

neon.write_vtu(mesh, "/tmp/cavity2d.vtu")
pv.read("/tmp/cavity2d.vtu").plot(show_edges=True)
