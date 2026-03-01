# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Read a CGNS mesh, write it to VTM, and visualize with pyvista."""

import os
from pathlib import Path

import neon
import pyvista as pv

# Locate the cavity mesh relative to the repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..", "..")
cgns_path = os.path.join(repo_root, "test", "mesh", "unstructured", "io", "meshFiles", "cavity2D.cgns")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

exec = neon.SerialExecutor()
mesh = neon.read_cgns(cgns_path, exec)

print(f"Cells: {mesh.n_cells}, Faces: {mesh.n_faces}")

output_path = str(output_dir / "cavity2d.vtm")
neon.write_vtm(mesh, output_path)
pv.read(output_path).plot(show_edges=True)
