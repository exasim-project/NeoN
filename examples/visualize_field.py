# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Write a uniform 2D mesh with scalar and vector VolumeFields to VTM and VTK HDF.

Each boundary patch receives a unique value so boundaries are visually distinct.
"""

from pathlib import Path

import neon

exec = neon.SerialExecutor()
mesh = neon.create_uniform_2d_mesh(exec, nx=4, ny=4)

print(f"Cells: {mesh.n_cells()}, Boundaries: {mesh.n_boundaries()}")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# --- Scalar field: pressure ---
pressure = neon.ScalarVolumeField(exec, "pressure", mesh)
neon.fill(pressure.internal_vector(), 0.0)
pressure.correct_boundary_conditions()

# Set a unique value on each boundary patch (patch i → value i+1)
bd = pressure.boundary_data()
for i in range(mesh.n_boundaries()):
    start, end = bd.range(i)
    neon.fill_range(bd.value(), float(i + 1), start, end)

# --- Vector field: velocity ---
velocity = neon.VectorVolumeField(exec, "velocity", mesh)
neon.fill(velocity.internal_vector(), neon.Vec3(0.0, 0.0, 0.0))
velocity.correct_boundary_conditions()

bd_v = velocity.boundary_data()
for i in range(mesh.n_boundaries()):
    start, end = bd_v.range(i)
    neon.fill_range(bd_v.value(), neon.Vec3(float(i + 1), 0.0, 0.0), start, end)

# Write both fields together — topology built once
with neon.MeshWriter(mesh, str(output_dir / "fields.vtm")) as w:
    w.add_field(pressure)
    w.add_field(velocity)

with neon.MeshWriter(mesh, str(output_dir / "fields.vtkhdf"), fmt="vtkhdf") as w:
    w.add_field(pressure)
    w.add_field(velocity)

print(f"Written: {output_dir}/fields.vtm, {output_dir}/fields.vtkhdf")
