# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Partition a 2D mesh across MPI ranks, sync global cell IDs over ghost cells, write to disc."""

import sys
import numpy as np
import neon
import pyvista as pv

# --- Initialize NeoN (Kokkos + MPI) ---
neon.initialize(sys.argv)

# --- MPI environment ---
mpi_env = neon.MPIEnvironment()
rank = mpi_env.rank()
n_ranks = mpi_env.size()

# --- Mesh creation & partitioning (every rank has the full mesh) ---
exec_ = neon.SerialExecutor()
mesh = neon.create_uniform_2d_mesh(exec_, nx=8, ny=8)
cell_part = neon.partition_mesh(mesh, n_ranks)

# --- Each rank extracts its own sub-mesh ---
sub = neon.extract_sub_mesh(mesh, cell_part, rank)
global_ids = neon.get_global_cell_ids(sub)
ghost_ids = neon.get_ghost_cell_ids(sub)
n_cells = sub.n_cells()
n_ghosts = len(ghost_ids)

print(f"[Rank {rank}] {n_cells} cells, {n_ghosts} ghosts")

# --- Create field: global cell ID as scalar, ghost cells = -1 ---
field = neon.ScalarVector(exec_, n_cells + n_ghosts, -1.0)
for i in range(n_cells):
    field[i] = float(global_ids[i])

# --- Write "before sync" mesh (ghost cells show -1) ---
vtm_before = f"rank{rank}_before_sync.vtm"
neon.write_vtm(sub, vtm_before, include_ghosts=True)

grid_before = pv.read(vtm_before)
internal_before = grid_before[0]
ids_before = np.array([field[i] for i in range(n_cells + n_ghosts)])
internal_before.cell_data["globalCellId"] = ids_before
internal_before.save(f"rank{rank}_before_sync.vtu")

# --- Build communicator from sub-mesh metadata ---
comm = neon.build_communicator(sub, mpi_env)

# --- Synchronize ghost cells ---
comm.start_comm(field, "sync_ids")
comm.finalise_comm(field, "sync_ids")

# --- Verify: ghost cells should now have correct global IDs ---
all_ok = True
for i in range(n_ghosts):
    expected = float(ghost_ids[i])
    actual = field[n_cells + i]
    if actual != expected:
        print(f"[Rank {rank}] MISMATCH ghost {i}: expected {expected}, got {actual}")
        all_ok = False

if all_ok:
    print(f"[Rank {rank}] Ghost sync verified OK")
else:
    print(f"[Rank {rank}] Ghost sync FAILED")
    sys.exit(1)

# --- Write "after sync" mesh (ghost cells show correct global IDs) ---
vtm_after = f"rank{rank}_after_sync.vtm"
neon.write_vtm(sub, vtm_after, include_ghosts=True)

grid_after = pv.read(vtm_after)
internal_after = grid_after[0]
ids_after = np.array([field[i] for i in range(n_cells + n_ghosts)])
internal_after.cell_data["globalCellId"] = ids_after
internal_after.save(f"rank{rank}_after_sync.vtu")

print(f"[Rank {rank}] Wrote rank{rank}_before_sync.vtu and rank{rank}_after_sync.vtu")

# --- Clean up Kokkos-managed objects before finalize ---
del field, comm, sub, mesh
import gc
gc.collect()

# --- Finalize NeoN (Kokkos + MPI) ---
neon.finalize()
