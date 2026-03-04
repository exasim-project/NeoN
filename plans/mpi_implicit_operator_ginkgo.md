# Plan: MPI Support for Implicit Operators with Ginkgo Distributed Solvers

## Summary Table

| Phase | What | Complexity | Key Changes | Depends On |
|-------|------|------------|-------------|------------|
| **1** | Global column indices in SparsityPattern | Medium | `SparsityPattern`, `FaceToMatrixAddress`, `extractSubMesh` | Mesh partitioning (done) |
| **2** | Enable `GINKGO_BUILD_MPI` in CMake | Low | `CxxThirdParty.cmake`, `CMakeLists.txt` | — |
| **3** | Create `DistributedLinearSystem` | High | New class wrapping `gko::experimental::distributed::Matrix` + `Vector` | Phase 1, 2 |
| **4** | Distributed assembly in FV operators | High | Laplacian, divergence, ddt operators produce global col indices for off-proc entries | Phase 1, 3 |
| **5** | Distributed Ginkgo solver | Medium | New `DistributedGinkgoSolver` or extend `GinkgoSolver` | Phase 2, 3 |
| **6** | Ghost cell sync in iterative solve loop | Medium | `dsl::solver.hpp`, `Communicator` integration | Phase 5, existing Communicator |
| **7** | Distributed preconditioner support | Medium | Schwarz wrapper, local preconditioner config | Phase 5 |
| **8** | Testing & validation | High | MPI-parallel Poisson solve, convergence checks | All phases |

---

## Current State

### What works

- **Mesh partitioning**: METIS-based `partitionMesh()` + `extractSubMesh()` produces sub-meshes with ghost cells, proc-boundary patches (`proc<p>to<q>`), and comm maps in `stencilDB`
- **Ghost cell sync**: `Communicator` class with non-blocking `startComm`/`finaliseComm` for `Vector<T>` exchange
- **Local implicit solve**: Full FV operator chain (`imp::ddt`, `imp::laplacian`, `imp::div`, `imp::source`) assembles into `LinearSystem` → Ginkgo solver (local CSR, zero-copy)
- **Ginkgo integration**: `GinkgoSolver` maps NeoN executors to Ginkgo executors, creates zero-copy CSR views, supports scalar and Vec3 systems
- **SparsityPattern**: CSR format with `rowOffs_`/`colIdxs_` using **local** indices only

### What is missing for distributed implicit solves

1. **Global column indices** — The current `SparsityPattern` uses local indices. For distributed solvers, off-diagonal entries (coupling to ghost cells on other ranks) need global column indices
2. **Ginkgo MPI disabled** — `GINKGO_BUILD_MPI OFF` in CMake; Ginkgo's `gko::experimental::distributed` namespace is not compiled
3. **No distributed matrix assembly** — FV operators fill a local `LinearSystem`; there is no mechanism to express inter-rank matrix entries
4. **No distributed solver** — `GinkgoSolver` creates local `gko::matrix::Csr` only; no `gko::experimental::distributed::Matrix` support
5. **No global-to-local index mapping** — No `gko::experimental::distributed::Partition` or `index_map` is constructed from the sub-mesh metadata
6. **No MPI-aware solve loop** — `dsl::solve()` does not synchronize ghost cells between outer iterations or provide global convergence checks

---

## Ginkgo Distributed API Overview

Ginkgo provides a complete distributed linear algebra stack under `gko::experimental::distributed`:

| Ginkgo Class | Purpose | NeoN Mapping |
|-------------|---------|--------------|
| `mpi::communicator` | Wraps `MPI_Comm` | `mpi::MPIEnvironment` |
| `Partition<LocalIdx, GlobalIdx>` | Maps global row range → rank ownership | Built from sub-mesh `nCells` per rank |
| `Matrix<ValueType, LocalIdx, GlobalIdx>` | Distributed CSR; splits into local + non-local parts automatically | Wraps NeoN's assembled `LinearSystem` |
| `Vector<ValueType>` | Distributed dense vector (row-partitioned) | Wraps NeoN's `Vector<T>` (rhs, solution) |
| `preconditioner::Schwarz` | Domain-decomposition preconditioner | Wraps local Ginkgo preconditioner per rank |
| Standard solvers (CG, GMRES, BiCGSTAB, ...) | Work transparently on distributed LinOps | Same config as local `GinkgoSolver` |

**Key workflow**:
```
1. Each rank assembles local rows with GLOBAL column indices (local cols + ghost cols)
2. Create gko::experimental::distributed::Partition from per-rank row counts
3. Call distributed::Matrix::read_distributed(matrix_data, partition)
   → Ginkgo automatically separates local vs non-local parts
   → Ginkgo handles all MPI communication in SpMV
4. Create distributed::Vector for rhs and solution
5. Use standard Ginkgo solver (CG, GMRES, etc.) — it dispatches to distributed apply()
6. Schwarz preconditioner applies local preconditioner per subdomain
```

---

## Implementation Plan

### Phase 1: Global Column Indices in SparsityPattern

**Goal**: Enable the sparsity pattern and matrix assembly to reference ghost cells using global column indices, so that Ginkgo's `read_distributed` can identify off-process couplings.

**Files**:
- `include/NeoN/linearAlgebra/sparsityPattern.hpp`
- `src/linearAlgebra/sparsityPattern.cpp`
- `src/linearAlgebra/faceToMatrixAddress.cpp`
- `src/mesh/unstructured/partition/extractSubMesh.cpp`

**Approach — Two-layer indexing**:

The current `SparsityPattern` stores `colIdxs_` as local indices `[0, nCells + nGhosts)`. For distributed assembly, we need a **global column index array** that maps ghost cell columns to their global IDs.

Option A — **Local-to-global column map** (recommended):
- Store an additional `Vector<globalIdx> localToGlobalCol_` in `SparsityPattern` (or in a separate `DistributedSparsityInfo` struct)
- For owned cells `[0, nCells)`, `localToGlobalCol_[i] = globalCellIds[i]`
- For ghost cells `[nCells, nCells+nGhosts)`, `localToGlobalCol_[i] = ghostCellGlobalIds[i - nCells]`
- This map is already available from `extractSubMesh`'s stencilDB (`partition::globalCellIds` + `partition::ghostCellGlobalIds`)
- The local CSR remains unchanged for local solvers; only the distributed solver path uses the global map

Option B — Store global colIdxs directly:
- Replace `colIdxs_` with global indices
- Breaks the local zero-copy Ginkgo path
- **Not recommended**

**Key data to store in stencilDB** (already partially done):
```
partition::globalCellIds       → vector<label>   (local cell → global cell)
partition::ghostCellGlobalIds  → vector<label>   (ghost cell → global cell)
partition::nGlobalCells        → label           (total cells in full mesh)
```

### Phase 2: Enable Ginkgo MPI Build

**Goal**: Compile Ginkgo with `GINKGO_BUILD_MPI=ON` when NeoN has MPI enabled.

**Files**:
- `cmake/CxxThirdParty.cmake`

**Change**:
```cmake
# Before:
"GINKGO_BUILD_MPI OFF"

# After:
"GINKGO_BUILD_MPI ${NeoN_ENABLE_MPI}"
```

This enables the `gko::experimental::distributed` namespace and `gko::experimental::mpi` wrappers. Ginkgo's MPI support is header-compatible with NeoN's existing MPI usage (both use `MPI_Comm`).

**Considerations**:
- Ginkgo's MPI layer is lightweight — it wraps `MPI_Comm` without owning init/finalize
- Ginkgo 1.11+ marks distributed as `experimental` but it is functionally stable
- This increases compile time slightly but only affects MPI builds

### Phase 3: `DistributedLinearSystem` Class

**Goal**: A new class that wraps the assembled local `LinearSystem` and provides the bridge to Ginkgo's distributed types.

**New file**: `include/NeoN/linearAlgebra/distributedLinearSystem.hpp`

**Design**:
```cpp
namespace NeoN::la {

template<typename ValueType, typename IndexType = localIdx>
class DistributedLinearSystem {
public:
    DistributedLinearSystem(
        const LinearSystem<ValueType, CSRMatrix<ValueType, IndexType>>& localSystem,
        const UnstructuredMesh& mesh,          // provides stencilDB with global IDs
        const mpi::MPIEnvironment& mpiEnv
    );

    // Build gko::experimental::distributed::Partition from per-rank cell counts
    // (MPI_Allgather of local nCells)
    std::shared_ptr<const gko::experimental::distributed::Partition<IndexType, globalIdx>>
    buildPartition() const;

    // Convert local CSR with local-to-global col mapping into
    // gko::matrix_data<ValueType, globalIdx> suitable for read_distributed()
    gko::matrix_data<ValueType, globalIdx> toGlobalMatrixData() const;

    // Accessors
    const LinearSystem<ValueType, CSRMatrix<ValueType, IndexType>>& localSystem() const;
    localIdx nLocalRows() const;
    globalIdx nGlobalRows() const;

private:
    const LinearSystem<ValueType, CSRMatrix<ValueType, IndexType>>& localSystem_;
    Vector<globalIdx> localToGlobalCol_;   // from Phase 1
    globalIdx nGlobalRows_;                // from MPI_Allreduce
    mpi::MPIEnvironment mpiEnv_;
};

} // namespace NeoN::la
```

**`toGlobalMatrixData()` implementation sketch**:
```cpp
gko::matrix_data<ValueType, globalIdx> toGlobalMatrixData() const {
    auto hostSys = localSystem_.copyToHost();
    auto [coeffs, sparsity] = hostSys.matrix().view();

    gko::matrix_data<ValueType, globalIdx> data;
    data.size = {nGlobalRows_, nGlobalRows_};

    for (localIdx row = 0; row < nLocalRows(); ++row) {
        globalIdx globalRow = localToGlobalCol_[row];
        for (auto j = sparsity.rowOffs[row]; j < sparsity.rowOffs[row + 1]; ++j) {
            globalIdx globalCol = localToGlobalCol_[sparsity.colIdxs[j]];
            data.nonzeros.emplace_back(globalRow, globalCol, coeffs[j]);
        }
    }
    return data;
}
```

**Note on boundary matrix**: The existing `LinearSystem` has a `boundaryMatrix_` that stores boundary condition contributions separately. In the distributed case, proc-boundary patches contribute off-diagonal entries to the **main** matrix (not the boundary matrix). The FV operators need to be aware of this distinction — see Phase 4.

### Phase 4: Distributed Assembly in FV Operators

**Goal**: Make the implicit FV operators (laplacian, div, ddt) correctly populate matrix entries for proc-boundary faces using global column indices.

**Problem**: Currently, proc-boundary patches are treated like regular boundary patches. The `computeLaplacianImpl` function only iterates over internal faces (owner-neighbour pairs within the sub-mesh). Ghost cell couplings from proc-boundary faces are **not** assembled into the matrix — they are handled by explicit ghost cell updates between outer iterations.

**Two approaches**:

#### Approach A: Assemble proc-boundary into matrix (full implicit coupling)

- For each proc-boundary face, add off-diagonal entry coupling owned cell → ghost cell
- The ghost cell column uses a local index that maps to a global index via `localToGlobalCol_`
- This gives Ginkgo the full distributed matrix, enabling true parallel implicit solves
- Requires modifying `FaceToMatrixAddress` to include proc-boundary face entries in the sparsity pattern

**Changes**:
- `src/linearAlgebra/faceToMatrixAddress.cpp`: Include proc-boundary faces when building sparsity
- `src/finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp`: Iterate over proc-boundary faces in `computeLaplacianImpl`
- `src/finiteVolume/cellCentred/operators/gaussGreenDiv.cpp`: Same for divergence
- The proc-boundary face has `owner` (local cell) and `neighbour` (ghost cell index = `nCells + ghostIdx`)
- The sparsity pattern row for the owned cell gets an extra column entry for the ghost cell

#### Approach B: Ghost-cell-based outer iteration (simpler, OpenFOAM-style)

- Keep the matrix purely local (only owned cells in rows and columns)
- Move ghost-cell contributions to the RHS explicitly: `rhs[owner] -= offDiagCoeff * ghostValue`
- Iterate: solve local system → sync ghost cells → re-assemble → solve again
- Convergence checked globally via `MPI_Allreduce` on residual norm

**Recommendation**: **Start with Approach B** for simplicity and compatibility with the existing local `GinkgoSolver`. Then implement **Approach A** as the advanced path using Ginkgo distributed solvers for cases where the coupling is strong (e.g., implicit pressure solvers).

**For Approach B — proc-boundary RHS contribution**:

The key insight is that proc-boundary faces behave like internal faces, but the neighbour cell lives on another rank. During assembly:

```cpp
// For each proc-boundary face f:
//   owner = local cell index
//   ghost = nCells + ghostLocalIdx (value available after ghost sync)
//
// Laplacian contribution:
//   A[owner][owner] += gamma * faceArea / dist   (diagonal — already done via internal faces)
//   rhs[owner]      -= gamma * faceArea / dist * ghostValue  (explicit correction)
```

This requires a post-assembly step that reads ghost cell values and adjusts the RHS. This step must be called after each ghost sync in the outer iteration loop.

### Phase 5: Distributed Ginkgo Solver

**Goal**: A solver that uses `gko::experimental::distributed::Matrix` and `Vector` for true parallel solves.

**New file**: `include/NeoN/linearAlgebra/distributedGinkgoSolver.hpp`

**Design** (for Approach A):
```cpp
namespace NeoN::la::ginkgo {

class DistributedGinkgoSolver : public SolverFactory::Register<DistributedGinkgoSolver> {
public:
    DistributedGinkgoSolver(Executor exec, const Dictionary& config, MPI_Comm comm);

    static std::string name() { return "DistributedGinkgo"; }

    SolverStats solve(
        const DistributedLinearSystem<scalar>& distSys,
        Vector<scalar>& x
    ) const;

private:
    std::shared_ptr<const gko::Executor> gkoExec_;
    gko::experimental::mpi::communicator gkoComm_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
};

} // namespace NeoN::la::ginkgo
```

**Solve workflow**:
```cpp
SolverStats DistributedGinkgoSolver::solve(
    const DistributedLinearSystem<scalar>& distSys,
    Vector<scalar>& x
) const {
    // 1. Build partition from per-rank sizes
    auto partition = distSys.buildPartition();

    // 2. Convert to Ginkgo matrix_data with global indices
    auto matData = distSys.toGlobalMatrixData();

    // 3. Create distributed matrix
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, localIdx, globalIdx>;
    auto A = dist_mtx::create(gkoExec_, gkoComm_);
    A->read_distributed(matData, partition);

    // 4. Create distributed vectors for rhs and x
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    // ... wrap local data into distributed vectors ...

    // 5. Generate solver and apply
    auto solver = factory_->generate(gko::share(A));
    solver->apply(b, gko_x);

    // 6. Copy solution back to NeoN vector
    // ...
}
```

**For Approach B** (simpler, extends existing `GinkgoSolver`):
- No new solver class needed
- The existing `GinkgoSolver` solves the local system as-is
- A wrapper in `dsl::solve` handles the outer iteration:
  ```
  while not converged:
      sync ghost cells (Communicator)
      assemble local system (with ghost RHS correction)
      solve local system (existing GinkgoSolver)
      check global convergence (MPI_Allreduce on residual)
  ```

### Phase 6: Ghost Cell Sync in the Iterative Solve Loop

**Goal**: Integrate ghost synchronization into the implicit solve workflow.

**Files**: `include/NeoN/dsl/solver.hpp`

**Changes to `iterativeSolveImpl`**:

```cpp
template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t, scalar dt,
    const Dictionary& fvSolution,
    Communicator* comm,  // NEW: optional communicator for MPI
    // ...
) {
    auto [sparsity, ls] = exp.assemble(solution.mesh(), t, dt, ps);

    // Approach B outer iteration:
    for (int outerIter = 0; outerIter < maxOuterIters; ++outerIter) {
        ls.reset();
        exp.assemble(t, dt, ls, ps);

        // Apply proc-boundary ghost correction to RHS
        if (comm) {
            applyProcBoundaryCorrection(ls, solution, mesh);
        }

        // Subtract explicit source
        auto expTmp = exp.explicitOperation(solution.mesh().nCells());
        // ... existing code ...

        auto solver = la::Solver(solution.exec(), fvSolution);
        auto stats = solver.solve(ls, solution.internalVector());

        // Sync ghost cells for next iteration
        if (comm) {
            comm->startComm(solution.internalVector(), "implicit_solve");
            comm->finaliseComm(solution.internalVector(), "implicit_solve");

            // Global convergence check
            scalar localResidual = stats.entries.back().finalResNorm;
            scalar globalResidual;
            MPI_Allreduce(&localResidual, &globalResidual, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (globalResidual < tolerance) break;
        } else {
            break;  // serial: single solve is sufficient
        }
    }
}
```

### Phase 7: Distributed Preconditioner Support

**Goal**: Enable Schwarz-type preconditioners that apply local preconditioners per subdomain.

**Ginkgo provides `gko::experimental::distributed::preconditioner::Schwarz`**:
- Takes a local solver/preconditioner factory
- Applies it to each rank's local portion
- No overlap communication needed (handled by the distributed matrix apply)

**Configuration example** (NeoN Dictionary → Ginkgo config):
```json
{
    "solver": "DistributedGinkgo",
    "type": "solver::Cg",
    "criteria": { ... },
    "preconditioner": {
        "type": "preconditioner::Schwarz",
        "local_solver": {
            "type": "preconditioner::Ilu",
            "factorization": { "type": "factorization::Ilu" }
        }
    }
}
```

For Approach B (local solves), the local preconditioner is just the normal Ginkgo preconditioner — no Schwarz wrapper needed since each rank solves independently.

### Phase 8: Testing & Validation

**Test cases**:

| Test | Description | Validates |
|------|-------------|-----------|
| `test_global_sparsity` | Build sub-mesh sparsity with global col indices, verify global IDs match | Phase 1 |
| `test_distributed_poisson_1d` | 1D Poisson with 2-4 ranks, compare to serial solution | Phase 4-6 |
| `test_distributed_laplacian_2d` | 2D uniform grid, `imp::laplacian(gamma, phi)`, verify convergence | Phase 4-6 |
| `test_ginkgo_distributed_matrix` | Create `distributed::Matrix` from NeoN data, verify SpMV | Phase 3, 5 |
| `test_schwarz_preconditioner` | Verify Schwarz-preconditioned CG converges in fewer iterations | Phase 7 |
| `test_ghost_sync_in_solve` | Outer iteration converges to same solution as serial | Phase 6 |

**Validation strategy**:
1. Solve identical problem in serial (1 rank, full mesh) and parallel (N ranks, partitioned)
2. Gather parallel solution to rank 0, compare field values (tolerance: solver convergence criterion)
3. Verify iteration counts are comparable (parallel may need more due to domain decomposition)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     dsl::solve()                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Outer iteration loop                     │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────┐  │  │
│  │  │ Ghost sync  │──▶│   Assemble   │──▶│  Solve    │  │  │
│  │  │(Communicator│   │ (FV Operators│   │ (local or │  │  │
│  │  │ startComm/  │   │  + procBC    │   │ distributed│ │  │
│  │  │ finaliseComm│   │  correction) │   │  Ginkgo)  │  │  │
│  │  └─────────────┘   └──────────────┘   └───────────┘  │  │
│  │         ▲                                    │        │  │
│  │         └───── global convergence check ─────┘        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

                    Approach A (full distributed)
                    ┌──────────────────────────┐
                    │ DistributedLinearSystem   │
                    │  ├─ localSystem (CSR)     │
                    │  ├─ localToGlobalCol map  │
                    │  └─ partition (per-rank)  │
                    └────────────┬─────────────┘
                                 │ toGlobalMatrixData()
                                 ▼
                    ┌──────────────────────────┐
                    │ gko::distributed::Matrix │
                    │  ├─ local part (auto)    │
                    │  ├─ non-local part (auto)│
                    │  └─ MPI comm patterns    │
                    └────────────┬─────────────┘
                                 │ solver->apply(b, x)
                                 ▼
                    ┌──────────────────────────┐
                    │ gko::solver::Cg/Gmres/.. │
                    │  + Schwarz preconditioner │
                    └──────────────────────────┘
```

---

## Recommended Phasing

### Milestone 1: Local solve + ghost iteration (Approach B)
- **Phase 1** (partial): Store `nGlobalCells` and global cell ID arrays in stencilDB
- **Phase 4B**: Implement `applyProcBoundaryCorrection()` for RHS fixup
- **Phase 6**: Outer iteration loop in `dsl::solve` with ghost sync + global convergence
- **Result**: Parallel implicit solves work with existing `GinkgoSolver`, no Ginkgo MPI needed

### Milestone 2: Full distributed Ginkgo solve (Approach A)
- **Phase 1** (full): `localToGlobalCol` mapping
- **Phase 2**: Enable `GINKGO_BUILD_MPI`
- **Phase 3**: `DistributedLinearSystem` class
- **Phase 5**: `DistributedGinkgoSolver`
- **Phase 7**: Schwarz preconditioner config
- **Result**: True distributed implicit solves, Ginkgo handles all MPI in SpMV

### Milestone 3: Production readiness
- **Phase 8**: Full test suite
- GPU-aware MPI (Ginkgo + CUDA/HIP aware MPI for direct GPU buffer transfer)
- Performance benchmarking vs Milestone 1

---

## Risks & Open Questions

1. **Ginkgo MPI maturity**: The `gko::experimental::distributed` API is marked experimental in Ginkgo 1.11. Need to verify stability with the pinned version (`49bff363`). The API may change in future Ginkgo releases.

2. **Zero-copy vs copy for distributed**: The local Ginkgo path uses zero-copy CSR views. The distributed path requires `matrix_data` → `read_distributed()` which copies data internally. This is a one-time cost per assembly but worth noting for performance.

3. **Proc-boundary vs internal face distinction**: Current `FaceToMatrixAddress` only processes internal faces. Adding proc-boundary faces to the sparsity pattern (Approach A) changes the nnz count and requires careful integration with existing `FaceToMatrixAddress::upperIdx`/`lowerIdx` logic.

4. **Boundary matrix handling**: The existing `LinearSystem` has a separate `boundaryMatrix_` for BC contributions. Proc-boundary entries must go into the **main** matrix (not `boundaryMatrix_`), since they represent real inter-cell coupling, not boundary conditions. Need clear distinction between physical BCs and proc BCs.

5. **Vec3 systems**: Ginkgo's distributed matrix supports scalar types. For `LinearSystem<Vec3>`, the current approach of unpacking to 3 scalar systems needs to work with the distributed path too. This triples the communication volume.

6. **Global index type**: NeoN uses `label` (int32) and `localIdx` (int32). For large meshes (>2B cells), a 64-bit global index type may be needed. Ginkgo supports `int64` as `global_index_type`.

7. **Convergence of Approach B**: The outer iteration (local solve + ghost sync) is equivalent to a block Jacobi method. For strongly coupled problems (e.g., pressure equation), this may converge slowly or not at all without under-relaxation. Approach A with Ginkgo distributed solver avoids this issue.

---

## File Change Summary

| File | Action | Phase | Description |
|------|--------|-------|-------------|
| `cmake/CxxThirdParty.cmake` | Modify | 2 | `GINKGO_BUILD_MPI ${NeoN_ENABLE_MPI}` |
| `include/NeoN/linearAlgebra/sparsityPattern.hpp` | Modify | 1 | Add optional `localToGlobalCol_` |
| `src/linearAlgebra/faceToMatrixAddress.cpp` | Modify | 1, 4A | Include proc-boundary faces in sparsity |
| `src/mesh/unstructured/partition/extractSubMesh.cpp` | Modify | 1 | Store `nGlobalCells` in stencilDB |
| `include/NeoN/linearAlgebra/distributedLinearSystem.hpp` | **New** | 3 | Bridge between NeoN and Ginkgo distributed |
| `include/NeoN/linearAlgebra/distributedGinkgoSolver.hpp` | **New** | 5 | Distributed Ginkgo solver wrapper |
| `src/linearAlgebra/distributedGinkgoSolver.cpp` | **New** | 5 | Implementation |
| `include/NeoN/dsl/solver.hpp` | Modify | 6 | Outer iteration loop with ghost sync |
| `src/finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp` | Modify | 4 | Proc-boundary face contributions |
| `src/finiteVolume/cellCentred/operators/gaussGreenDiv.cpp` | Modify | 4 | Proc-boundary face contributions |
| `test/linearAlgebra/distributedGinkgo.cpp` | **New** | 8 | MPI integration tests |

---

## References

- [Ginkgo distributed Matrix API](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Matrix.html)
- [Ginkgo distributed Vector API](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Vector.html)
- [Ginkgo distributed Partition API](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/classgko_1_1experimental_1_1distributed_1_1Partition.html)
- [Ginkgo Schwarz preconditioner](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/namespacegko_1_1experimental_1_1distributed_1_1preconditioner.html)
- [Ginkgo distributed-solver example](https://github.com/ginkgo-project/ginkgo/blob/develop/examples/distributed-solver/distributed-solver.cpp)
- [Towards Distributed Linear Solvers on GPUs using Ginkgo (OpenFOAM Workshop 2022)](https://www.keysight.com/cae/sites/default/files/resource/other/3280/30_Abstract_OpenFOAM_2022_Olenik_KIT.pdf)
