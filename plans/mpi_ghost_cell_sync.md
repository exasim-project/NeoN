# Plan: MPI Ghost Cell Synchronization for Python

## Goal

Enable a Python example (run via `mpirun -np N python example.py`) that:

1. Creates a global mesh on every rank
2. Partitions it into N parts (one per MPI rank)
3. Each rank extracts its own sub-mesh (with ghost cells)
4. Creates a field (e.g. global cell IDs) over real + ghost cells
5. Synchronizes ghost cell values across ranks using MPI
6. Visualizes the result (each partition colored by global cell ID, ghost cells correctly filled)

---

## Summary

| Phase | What | Key Files | Status |
|-------|------|-----------|--------|
| **1** | Store comm send/receive maps in `extractSubMesh` stencilDB | `extractSubMesh.cpp` | TODO |
| **2** | Add MPI init/finalize to existing `initialize`/`finalize` | `initialization.hpp`, `initialization.cpp` | TODO |
| **3** | Bind `MPIEnvironment` to Python | `mpi.cpp` (new) | TODO |
| **4** | Bind `Communicator` + `build_communicator` helper | `mpi.cpp` (new) | TODO |
| **5** | Ensure Vector supports ghost-sized creation & indexed access | `vectors.cpp` | TODO |
| **6** | Python example: MPI ghost cell sync + visualization | `examples/mpi_ghost_sync.py` (new) | TODO |

---

## Current State

### What exists

- **Mesh partitioning**: `partitionMesh()` (METIS) + `extractSubMesh()` produce sub-meshes with ghost cells, proc boundary patches (`proc<p>to<q>`), and stencilDB metadata
- **Ghost cell data in stencilDB**: `partition::globalCellIds`, `partition::ghostCellGlobalIds`, `partition::ghostCellVolumes`, `partition::ghostCellCentres`, `partition::ghostCellFaceNodes`, `partition::ghostPoints`
- **MPI C++ infrastructure**: `MPIInit`, `MPIEnvironment`, `HalfDuplexCommBuffer`, `FullDuplexCommBuffer`, `Communicator` (all behind `#ifdef NF_WITH_MPI_SUPPORT`)
- **Communicator API**: `startComm(Vector<T>&, commName)` / `isComplete()` / `finaliseComm(Vector<T>&, commName)` with `CommMap` (per-rank send/receive local indices)
- **Python bindings**: nanobind-based, covering mesh creation, partitioning, extraction, ghost cell ID queries, VTM writing
- **Existing Vector bindings**: `ScalarVector`, `VectorVector`, `LabelVector` exposed to Python

### What is missing

1. **CommMap construction** — `extractSubMesh()` does not store the per-neighbor-partition send/receive index mapping needed by `Communicator`
2. **MPI Python bindings** — `MPIInit`, `MPIEnvironment` are not exposed to Python
3. **Communicator Python bindings** — `Communicator`, `CommMap` not exposed to Python
4. **Helper to build Communicator from sub-mesh** — no convenience function exists
5. **Vector ghost cell support** — need a way to create a `Vector<T>` with `nCells + nGhosts` slots and fill it from Python

---

## Implementation Plan

### Phase 1: Store Communication Maps in `extractSubMesh`

**Files**: `src/mesh/unstructured/partition/extractSubMesh.cpp`

During extraction, `extractSubMesh` already has all information needed to build the `CommMap`:

- **Send map**: For each neighbor partition `q`, the unique local cell indices that own `proc<p>to<q>` boundary faces. These are the cells whose values rank `q` needs as ghost data.
- **Receive map**: For each neighbor partition `q`, the local ghost-cell indices (offset by `nSubCells`) for ghost cells belonging to partition `q`.

Add to stencilDB at the end of `extractSubMesh`:

```
"partition::nParts"          -> int (total number of partitions)
"partition::partId"          -> int (this partition's ID)
"partition::commSendMap"     -> shared_ptr<vector<vector<localIdx>>>
                                commSendMap[q] = list of local cell indices to send to rank q
                                (empty for non-neighbor ranks)
"partition::commReceiveMap"  -> shared_ptr<vector<vector<localIdx>>>
                                commReceiveMap[q] = list of local indices (in ghost region,
                                offset by nSubCells) to write received data into
                                (empty for non-neighbor ranks)
```

**Implementation detail**:
- After building `procFacesByNeighbor`, iterate over each `(neighborPartId, faceIndices)`:
  - Collect unique `pf.subOwner` values for sendMap[neighborPartId]
  - For each face's `pf.globalOther`, look up `ghostG2L[pf.globalOther]` + nSubCells for receiveMap[neighborPartId]
- Initialize both maps to `nParts` empty vectors
- Order matters: send and receive entries must correspond 1:1 (the cell sent at position i must match the ghost cell received at position i on the remote rank)

**Ordering constraint**: For correct MPI exchange, the send order on rank p for rank q must match the receive order on rank q for rank p. Since both ranks process proc faces from the same global mesh, we can enforce consistent ordering by sorting send/receive entries by the **global cell ID** of the cell being communicated. This ensures rank p sends cells in the same order that rank q expects to receive them.

### Phase 2: Add MPI to `initialize` / `finalize`

**Files**: `include/NeoN/core/initialization.hpp`, `src/bindings/initialization.cpp`

MPI init/finalize becomes part of the existing `neon.initialize()` / `neon.finalize()` lifecycle — no separate `MPIInit` object needed in Python.

#### C++ side (`initialization.hpp`)
```cpp
inline void initialize(int argc, char* argv[])
{
#ifdef NF_WITH_MPI_SUPPORT
    MPI_Init(&argc, &argv);  // or MPI_Init_thread if thread support enabled
#endif
    Kokkos::initialize(argc, argv);
    Logging::setNeonDefaultPattern();
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
#ifdef NF_WITH_MPI_SUPPORT
    MPI_Finalize();
#endif
}
```

- Guard with `MPI_Initialized()` check so it plays nicely with mpi4py (if MPI is already initialized, skip `MPI_Init`)
- Guard `MPI_Finalize()` with `MPI_Finalized()` check for the same reason
- The existing Python bindings (`initialization.cpp`) already handle argc/argv conversion, so no changes needed there

#### Python usage (unchanged API)
```python
import sys
import neon

neon.initialize(sys.argv)   # now also does MPI_Init when built with MPI
# ... work ...
neon.finalize()              # now also does MPI_Finalize
```

### Phase 3: Bind `MPIEnvironment` to Python

**New file**: `src/bindings/mpi.cpp`
**Modified files**: `src/bindings/bindings.hpp`, `src/bindings/neon.cpp`, `src/bindings/CMakeLists.txt`

```python
mpi_env = neon.MPIEnvironment()  # wraps MPI_COMM_WORLD
rank = mpi_env.rank()
size = mpi_env.size()
```

- Entire file guarded with `#ifdef NF_WITH_MPI_SUPPORT`; when MPI is disabled, `registerMPI` is a no-op
- No mpi4py interop needed for v1 — NeoN manages MPI lifecycle through `initialize`/`finalize`

### Phase 4: Bind Communicator + `build_communicator` Helper

**File**: `src/bindings/mpi.cpp` (same new file)

#### build_communicator helper
A Python-callable function that constructs a `Communicator` from a sub-mesh's stencilDB:

```python
comm = neon.build_communicator(sub_mesh, mpi_env)
```

C++ implementation:
```cpp
Communicator buildCommunicatorFromMesh(
    const UnstructuredMesh& mesh,
    const mpi::MPIEnvironment& mpiEnviron
) {
    auto nParts = *mesh.stencilDB().get<...>("partition::nParts");
    auto& sendData = *mesh.stencilDB().get<...>("partition::commSendMap");
    auto& recvData = *mesh.stencilDB().get<...>("partition::commReceiveMap");

    CommMap sendMap(nParts), receiveMap(nParts);
    for (int r = 0; r < nParts; ++r) {
        for (auto idx : sendData[r])
            sendMap[r].push_back(NodeCommMap{.local_idx = idx});
        for (auto idx : recvData[r])
            receiveMap[r].push_back(NodeCommMap{.local_idx = idx});
    }
    return Communicator(mpiEnviron, sendMap, receiveMap);
}
```

#### Communicator class binding
```python
comm.start_comm(vector, "tag")
comm.is_complete("tag")
comm.finalise_comm(vector, "tag")
```

Template instantiation for `scalar` (and optionally `label`/`Vec3`).

### Phase 5: Vector Ghost-Aware Creation

**File**: `src/bindings/vectors.cpp` (or `partition.cpp`)

Add a helper to create a vector sized for real + ghost cells and optionally pre-filled:

```python
# Create vector with nCells + nGhosts elements
field = neon.ScalarVector(exec, n_cells + n_ghosts)

# Or: convenience function
field = neon.create_ghost_field(sub_mesh, exec, fill_value=0.0)
```

Also need indexed write access to fill real cell values:
```python
field[i] = value  # already available via Vector.__setitem__
```

Check that `ScalarVector.__getitem__` and `__setitem__` are already bound. If not, add them.

### Phase 6: Python Example

**New file**: `examples/mpi_ghost_sync.py`

```python
"""Partition a 2D mesh across MPI ranks, sync global cell IDs over ghost cells, visualize."""

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
field = neon.ScalarVector(exec_, n_cells + n_ghosts)
for i, gid in enumerate(global_ids):
    field[i] = float(gid)
for i in range(n_ghosts):
    field[n_cells + i] = -1.0

# --- Build communicator from sub-mesh metadata ---
comm = neon.build_communicator(sub, mpi_env)

# --- Synchronize ghost cells ---
comm.start_comm(field, "sync_ids")
comm.finalise_comm(field, "sync_ids")

# --- Verify: ghost cells should now have correct global IDs ---
for i in range(n_ghosts):
    expected = float(ghost_ids[i])
    actual = field[n_cells + i]
    assert actual == expected, f"[Rank {rank}] ghost {i}: expected {expected}, got {actual}"

print(f"[Rank {rank}] Ghost sync verified OK")

# --- Write VTM with ghost cells and the synced field ---
vtm_path = f"partition_{rank}.vtm"
neon.write_vtm(sub, vtm_path, include_ghosts=True)

# --- Visualization (rank 0 only) ---
if rank == 0:
    plotter = pv.Plotter()
    grid = pv.read(vtm_path)
    internal = grid[0]

    # Attach the synced field as cell data
    all_ids = [field[i] for i in range(n_cells + n_ghosts)]
    internal.cell_data["globalCellId"] = np.array(all_ids)

    plotter.add_mesh(
        internal,
        scalars="globalCellId",
        show_edges=True,
        label=f"Rank 0",
    )
    plotter.add_legend()
    plotter.show()

# --- Finalize NeoN (Kokkos + MPI) ---
neon.finalize()
```

Run with: `mpirun -np 4 python examples/mpi_ghost_sync.py`

---

## Build System Changes

### CMakeLists.txt changes
- Ensure `NeoN_ENABLE_MPI=ON` is propagated to the bindings build
- Add `mpi.cpp` to the nanobind module sources in `src/bindings/CMakeLists.txt`
- The `NF_WITH_MPI_SUPPORT` compile definition should already flow through from the NeoN target

### Dependencies
- MPI must be available at build time (`find_package(MPI REQUIRED)`)
- At runtime: `mpirun` or equivalent launcher
- Optional: `mpi4py` for Python-side MPI interop (not required for v1)

---

## Testing Strategy

### C++ tests
- **Unit test for CommMap storage**: Verify that after `extractSubMesh`, the stencilDB contains correct `commSendMap` and `commReceiveMap` entries
- **Ordering test**: Verify send/receive ordering is consistent across partitions (can test with 2+ partitions in serial by checking both sides)

### Python tests (MPI)
- **test_mpi_ghost_sync.py**: Run with `mpirun -np 2` (or 4)
  - Create mesh, partition, extract sub-meshes
  - Fill field with global cell IDs
  - Sync ghost cells
  - Assert all ghost cells have correct values
- Use pytest-mpi or a simple subprocess-based runner

### Python tests (serial, no MPI)
- Verify that the stencilDB comm map entries exist and have correct sizes
- Verify that `build_communicator` raises a clear error when MPI is not available

---

## File Change Summary

| File | Action | Phase | Description |
|------|--------|-------|-------------|
| `src/mesh/unstructured/partition/extractSubMesh.cpp` | Modify | 1 | Store `commSendMap`, `commReceiveMap`, `nParts`, `partId` in stencilDB |
| `include/NeoN/core/initialization.hpp` | Modify | 2 | Add `MPI_Init`/`MPI_Finalize` to `initialize()`/`finalize()` |
| `src/bindings/mpi.cpp` | **New** | 3-4 | Bind `MPIEnvironment`, `Communicator`, `build_communicator` |
| `src/bindings/bindings.hpp` | Modify | 3 | Add `registerMPI()` declaration |
| `src/bindings/neon.cpp` | Modify | 3 | Call `registerMPI(m)` |
| `src/bindings/CMakeLists.txt` | Modify | 3 | Add `mpi.cpp` to sources |
| `examples/mpi_ghost_sync.py` | **New** | 6 | MPI ghost cell sync example |
| `test/bindings/test_mpi_ghost_sync.py` | **New** | 6 | MPI Python test |
| `test/mesh/unstructured/partition/partitionMesh.cpp` | Modify | 1 | Add comm map stencilDB tests |

---

## Risks & Open Questions

1. **Ordering consistency**: The send/receive map ordering must match across ranks. Sorting by global cell ID during construction in `extractSubMesh` should guarantee this, but needs careful testing.

2. **mpi4py interop**: If the user imports mpi4py before calling `neon.initialize()`, MPI will already be initialized. The `MPI_Initialized()` guard in Phase 2 handles this — `initialize()` skips `MPI_Init` if MPI is already active. Similarly `finalize()` checks `MPI_Finalized()`.

3. **Vector template instantiation**: The `Communicator::startComm<T>` and `finaliseComm<T>` are templates. We need to instantiate for `scalar` at minimum. If `label` or `Vec3` sync is needed, add those instantiations too.

4. **GPU executor**: The current Communicator accesses `field(idx)` which works for SerialExecutor. For GPU executors, data would need to be copied to host first. Out of scope for v1 (SerialExecutor only).

5. **Scalability**: Every rank creates the full global mesh and partitions it. This works for small meshes but won't scale. For v1 this is acceptable; a distributed mesh read is a future enhancement.
