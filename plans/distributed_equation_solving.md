# Distributed Equation Solving — Integration Plan

## Goal

Run FV operators (`laplacian`, `div`, `ddt`) assembled into a `LinearSystem` and solve the system in parallel across MPI ranks using Ginkgo's distributed solver.

## Current State

| Component | Status |
|-----------|--------|
| Mesh partitioning (`extractSubMesh` + METIS) | Done |
| Ghost cell exchange (`Communicator`) | Done |
| `GINKGO_BUILD_MPI` enabled | Done |
| Distributed Ginkgo test (raw Ginkgo types) | Done |
| Distributed Ginkgo test (NeoN `CSRMatrix` → Ginkgo) | Done |
| MPI `ctest` runner fix (`$<TARGET_FILE:>`) | Done |
| Proc-boundary patches created in `extractSubMesh` | Done |
| `stencilDB` stores `globalCellIds`, `ghostCellGlobalIds`, comm maps | Done |
| **FV operator assembly with ghost columns** | **TODO** |
| **Local-to-global index conversion in solver** | **TODO** |
| **Distributed solver integration** | **TODO** |

## Architecture

```
                     Serial (current)                     Distributed (target)
                     ────────────────                     ──────────────────────
  Mesh             → UnstructuredMesh                   → Partitioned SubMesh + ghost cells
                                                          stencilDB: globalCellIds, ghostGlobalIds
                     │                                    │
  Sparsity         → FaceToMatrixAddress                → Extended: includes proc-boundary faces
                     nCells rows, nCells cols              nCells rows, (nCells+nGhosts) cols
                     │                                    │
  Assembly         → Operator loops internal faces      → Same + proc-boundary face loop
                     + boundary faces (Robin BCs)         ghost columns for proc-neighbor coupling
                     │                                    │
  LinearSystem     → matrix_ (local CSR)                → matrix_ (local CSR with ghost columns)
                     rhs_, boundaryMatrix_                rhs_, boundaryMatrix_
                     │                                    │
  Solve            → GinkgoSolver (local CSR)           → local→global index mapping
                                                        → Ginkgo distributed::Matrix
                                                        → CG + Schwarz(Jacobi)
                                                          │
  Post-solve       → (none)                             → ghost sync of solution
```

## Phases

### Phase 1: Extend `FaceToMatrixAddress` for ghost columns

**Why**: Currently, `FaceToMatrixAddress` only creates sparsity entries for internal faces (both owner and neighbor are owned cells). Proc-boundary faces are treated as boundary faces with `calculated` BC (zero matrix contribution). To get implicit coupling across ranks, proc-boundary faces must create off-diagonal entries where the column index references a ghost cell.

**Files**:
- `src/linearAlgebra/faceToMatrixAddress.cpp` — `createSparsityPatternFaceToMatrixAddress()`
- `include/NeoN/linearAlgebra/faceToMatrixAddress.hpp`

**Changes**:
1. Accept an optional flag or mesh query to detect proc-boundary patches
2. For each proc-boundary face: add an off-diagonal entry in the owner row with column = ghost cell index (`nCells + ghostLocalIdx`)
3. Non-zero count becomes: `nCells + 2*nInternalFaces + nProcBoundaryFaces`
4. Sparsity pattern columns range: `[0, nCells + nGhosts)`

**Key detail**: The mesh's `BoundaryMesh` already has proc-boundary patches (named `proc0to1`, etc.) with `faceCells()` pointing to owned cells. Ghost cell indices are available from `stencilDB["partition::ghostCellGlobalIds"]`. Need a mapping from proc-boundary face → ghost cell local index. This can be built from the comm maps in stencilDB.

**Test**: Create a 2-rank partitioned 1D mesh, build `FaceToMatrixAddress`, verify sparsity includes ghost column entries.

### Phase 2: Assemble operators with ghost coupling

**Why**: The laplacian/div operators loop over internal faces and boundary faces. With proc-boundary faces now in the sparsity pattern, the operator assembly needs to fill the ghost-column off-diagonal entries.

**Files**:
- `src/finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp`
- (later) `src/finiteVolume/cellCentred/operators/gaussGreenDiv.cpp`

**Changes**:
Two approaches (can be done incrementally):

**Approach A — Treat proc-boundary faces like internal faces**:
- In the internal face loop, proc-boundary faces already have `owner` (owned cell) and `neighbour` (ghost cell index)
- If `extractSubMesh` sets proc-boundary faces as internal faces with ghost neighbors, the existing loop works unchanged
- Requires `extractSubMesh` to emit proc-boundary faces as internal faces (not boundary faces)
- Ghost cell data (centres, volumes) needed for `deltaCoeffs` etc.

**Approach B — Add separate proc-boundary loop in operators**:
- Keep proc-boundary as boundary faces
- Add a new loop that fills ghost-column off-diagonal entries using `FaceToMatrixAddress`
- Uses `bndCn` (ghost cell centres) from the boundary mesh for gradient computation
- More explicit, less invasive

**Recommended**: Start with **Approach B** — it doesn't change mesh structure.

**Test**: Partition a simple mesh, assemble laplacian, verify matrix entries match a reference serial assembly.

### Phase 3: Local-to-global index conversion

**Why**: Ginkgo's distributed solver needs global row/column indices. NeoN's assembled `LinearSystem` uses local indices.

**Files**:
- New: `include/NeoN/linearAlgebra/utilities.hpp` or extend `ginkgo.hpp`

**Function signature**:
```cpp
gko::matrix_data<scalar, label> toGlobalMatrixData(
    const CSRMatrix<scalar, localIdx>& matrix,
    std::span<const label> globalCellIds,      // [0, nCells) → global
    std::span<const label> ghostCellGlobalIds,  // [0, nGhosts) → global
    label nGlobalCells,
    label nLocalCells
);
```

**Logic** (already proven in `distributedGinkgo.cpp` test):
```cpp
for each row i in [0, nLocalCells):
    globalRow = globalCellIds[i]
    for each entry j in CSR row i:
        localCol = colIdxs[j]
        globalCol = (localCol < nLocalCells)
            ? globalCellIds[localCol]
            : ghostCellGlobalIds[localCol - nLocalCells]
        output.emplace_back(globalRow, globalCol, values[j])
```

**Test**: Unit test converting a known local CSR with ghost columns to global matrix_data.

### Phase 4: Distributed GinkgoSolver

**Why**: Need to plug the local→global conversion and distributed Ginkgo types into the solver flow.

**Two options**:

**Option A — Extend existing `GinkgoSolver`**:
- Add a `solveDistributed()` method
- Detect distributed context from mesh metadata
- Pro: single solver class, automatic dispatch
- Con: couples MPI into existing solver

**Option B — New `DistributedGinkgoSolver` class** (recommended):
- Registered as `"DistributedGinkgo"` in `SolverFactory`
- Takes `LinearSystem` + mesh reference
- Performs local→global conversion, creates distributed matrix/vectors, solves

**Solver workflow**:
```
1. Extract globalCellIds, ghostCellGlobalIds, nGlobalCells from mesh.stencilDB()
2. Convert LinearSystem.matrix() → gko::matrix_data (global indices)
3. Build Partition from global cell ID ranges (MPI_Allgather of local sizes)
4. Create distributed::Matrix via read_distributed()
5. Create distributed::Vector for rhs and x
6. Solve with CG + Schwarz(Jacobi) preconditioner
7. Copy local solution back into NeoN Vector
```

**Files**:
- New: `include/NeoN/linearAlgebra/distributedGinkgoSolver.hpp`
- New: `src/linearAlgebra/distributedGinkgoSolver.cpp`

**Test**: Partition a 1D mesh, assemble laplacian with ghost coupling, solve with `DistributedGinkgoSolver`, verify against analytic solution.

### Phase 5: Integrate into `dsl::solve()`

**Why**: The DSL layer needs to orchestrate ghost sync + assembly + distributed solve.

**Files**:
- `include/NeoN/dsl/solver.hpp` — `iterativeSolveImpl()`

**Changes to `iterativeSolveImpl()`**:
```cpp
// 1. Sync ghost cells with current field values
if (mesh.hasPartitionData()) {
    mesh.communicator().startComm(solution.internalVector(), "implicit");
    mesh.communicator().finaliseComm(solution.internalVector(), "implicit");
}

// 2. Assemble (operators now fill ghost-column entries)
auto [sparsity, ls] = exp.assemble(mesh, t, dt, ps);

// 3. Solve (DistributedGinkgoSolver auto-detects distributed context)
auto solver = la::Solver(solution.exec(), fvSolution);
return solver.solve(ls, solution.internalVector());
```

**Test**: End-to-end test: partition mesh → create volume field → assemble laplacian expression → dsl::solve → verify.

### Phase 6: Outer iteration for nonlinear / coupled systems

**Why**: For nonlinear problems or when boundary conditions depend on ghost values, multiple assembly-solve cycles may be needed.

**Changes**:
- Wrap Phase 5 in an outer loop with ghost sync between iterations
- Global convergence check via `MPI_Allreduce` on residual norms
- Configurable via `fvSolution` dictionary: `{"outerIterations": 10, "outerTolerance": 1e-6}`

**This phase is optional for the initial integration.**

## Implementation Order

```
Phase 1 (FaceToMatrixAddress)  ←── Foundation, unlocks everything
    │
    ├── Phase 2 (Operator assembly)  ←── Can test with reference serial solution
    │       │
    │       └── Phase 3 (Index conversion)  ←── Extraction of existing test code
    │               │
    │               └── Phase 4 (DistributedGinkgoSolver)  ←── Core integration
    │                       │
    │                       └── Phase 5 (dsl::solve)  ←── End-to-end
    │                               │
    │                               └── Phase 6 (Outer iteration)  ←── Optional
    │
    └── Can be tested independently at each phase
```

## Key Files Reference

| File | Role |
|------|------|
| `src/linearAlgebra/faceToMatrixAddress.cpp` | Sparsity pattern generation from mesh faces |
| `src/finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp` | Laplacian assembly (internal + boundary face loops) |
| `src/mesh/unstructured/partition/extractSubMesh.cpp` | Mesh partitioning, ghost cell creation, stencilDB population |
| `include/NeoN/linearAlgebra/linearSystem.hpp` | LinearSystem class (matrix + rhs + boundary) |
| `include/NeoN/linearAlgebra/ginkgo.hpp` | Current GinkgoSolver (local solves) |
| `include/NeoN/dsl/solver.hpp` | DSL solve flow (`iterativeSolveImpl`) |
| `include/NeoN/dsl/expression.hpp` | Expression assembly orchestration |
| `include/NeoN/mesh/unstructured/communicator.hpp` | Ghost cell MPI exchange |
| `test/linearAlgebra/distributedGinkgo.cpp` | Reference: local-to-global conversion pattern |

## Risks

| Risk | Mitigation |
|------|------------|
| `FaceToMatrixAddress` changes break existing serial tests | Keep changes additive — ghost columns only added when partition data exists |
| Ginkgo `read_distributed` performance (copies data) | Acceptable for first iteration; optimize with zero-copy later |
| Ghost cell geometric data (`deltaCoeffs`, `weights`) incorrect | `extractSubMesh` already computes `bndCn` from ghost centres; validate against serial |
| Non-uniform partitions (METIS) complicate Partition building | Use `Partition::build_from_contiguous` with actual ranges instead of `build_from_global_size_uniform` |
