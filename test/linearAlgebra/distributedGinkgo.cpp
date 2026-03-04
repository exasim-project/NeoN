// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/mesh/unstructured/partition/partitionMesh.hpp"
#include "NeoN/mesh/unstructured/partition/extractSubMesh.hpp"
#include "NeoN/mesh/unstructured/communicator.hpp"
#include "NeoN/linearAlgebra/distributedGinkgoSolver.hpp"

using NeoN::scalar;
using NeoN::label;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::SparsityPattern;
using NeoN::la::CSRMatrix;

namespace fvcc = NeoN::finiteVolume::cellCentred;

// Helper: build proc-boundary + fixedValue BCs for a partitioned sub-mesh
std::vector<fvcc::VolumeBoundary<scalar>> makePartitionedBCs(const NeoN::UnstructuredMesh& subMesh)
{
    const auto& patchNames =
        *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
    std::vector<fvcc::VolumeBoundary<scalar>> bcs;
    for (localIdx patchID = 0; patchID < subMesh.nBoundaries(); ++patchID)
    {
        const auto& pName = patchNames[static_cast<std::size_t>(patchID)];
        if (pName.substr(0, 4) == "proc")
        {
            NeoN::Dictionary d({{"type", std::string("procBoundary")}});
            bcs.emplace_back(subMesh, d, patchID);
        }
        else
        {
            NeoN::Dictionary d({{"type", std::string("fixedValue")}, {"fixedValue", scalar(0.0)}});
            bcs.emplace_back(subMesh, d, patchID);
        }
    }
    return bcs;
}

TEST_CASE("Distributed Ginkgo Solve - 1D Poisson")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = mpiEnv.rank();
    auto nRanks = mpiEnv.sizeRank();

    auto exec = gko::ReferenceExecutor::create();
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);

    // N-point 1D Poisson: -u'' = 1 on [0,1], u(0)=u(1)=0
    // Analytic solution: u(x) = x(1-x)/2
    const label nGlobal = 100;

    using part_type = gko::experimental::distributed::Partition<label, label>;
    auto partition = gko::share(
        part_type::build_from_global_size_uniform(exec, static_cast<label>(nRanks), nGlobal)
    );

    auto rangeStart = partition->get_range_bounds()[rank];
    auto rangeEnd = partition->get_range_bounds()[rank + 1];
    gko::matrix_data<scalar, label> matData {
        gko::dim<2> {static_cast<std::size_t>(nGlobal), static_cast<std::size_t>(nGlobal)}
    };
    gko::matrix_data<scalar, label> rhsData {gko::dim<2> {static_cast<std::size_t>(nGlobal), 1}};
    scalar h = 1.0 / (nGlobal + 1);

    for (label i = rangeStart; i < rangeEnd; ++i)
    {
        matData.nonzeros.emplace_back(i, i, 2.0 / (h * h));
        if (i > 0)
        {
            matData.nonzeros.emplace_back(i, i - 1, -1.0 / (h * h));
        }
        if (i < nGlobal - 1)
        {
            matData.nonzeros.emplace_back(i, i + 1, -1.0 / (h * h));
        }
        rhsData.nonzeros.emplace_back(i, 0, 1.0);
    }

    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    auto A = gko::share(dist_mtx::create(exec, comm));
    A->read_distributed(matData, partition);

    auto b = dist_vec::create(exec, comm);
    b->read_distributed(rhsData, partition);

    auto x = dist_vec::create(exec, comm);
    gko::matrix_data<scalar, label> zeroData {gko::dim<2> {static_cast<std::size_t>(nGlobal), 1}};
    x->read_distributed(zeroData, partition);

    using cg = gko::solver::Cg<scalar>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<scalar, label, label>;
    using jacobi = gko::preconditioner::Jacobi<scalar, label>;

    auto solver = cg::build()
                      .with_preconditioner(schwarz::build().with_local_solver(jacobi::build()))
                      .with_criteria(
                          gko::stop::Iteration::build().with_max_iters(200u),
                          gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(1e-10)
                      )
                      .on(exec)
                      ->generate(A);

    solver->apply(b, x);

    auto localVec = x->get_local_vector();
    for (label i = rangeStart; i < rangeEnd; ++i)
    {
        label localRow = i - rangeStart;
        scalar xi = static_cast<scalar>(i + 1) * h;
        scalar expected = xi * (1.0 - xi) / 2.0;
        scalar computed = localVec->at(localRow, 0);
        REQUIRE(computed == Catch::Approx(expected).margin(1e-6));
    }
}

TEST_CASE("Distributed Ginkgo Solve - NeoN LinearSystem")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = mpiEnv.rank();
    auto nRanks = mpiEnv.sizeRank();
    NeoN::Executor exec = NeoN::SerialExecutor();

    const label nGlobal = 100;
    const label nLocal = nGlobal / static_cast<label>(nRanks);
    const label rangeStart = static_cast<label>(rank) * nLocal;
    scalar h = 1.0 / (nGlobal + 1);

    // Determine ghost cells (left and right neighbors)
    label nGhosts = 0;
    label leftGhostLocal = -1;
    label rightGhostLocal = -1;
    std::vector<label> ghostCellGlobalIds;

    if (rank > 0)
    {
        leftGhostLocal = nLocal + nGhosts;
        ghostCellGlobalIds.push_back(rangeStart - 1);
        nGhosts++;
    }
    if (rank < static_cast<label>(nRanks) - 1)
    {
        rightGhostLocal = nLocal + nGhosts;
        ghostCellGlobalIds.push_back(rangeStart + nLocal);
        nGhosts++;
    }

    // Build local-to-global mapping for owned cells
    std::vector<label> globalCellIds(static_cast<std::size_t>(nLocal));
    std::iota(globalCellIds.begin(), globalCellIds.end(), rangeStart);

    // Build CSR sparsity with local indices (owned columns + ghost columns)
    std::vector<localIdx> colIdxs;
    std::vector<localIdx> rowOffs;
    std::vector<scalar> values;
    std::vector<scalar> rhsValues;
    rowOffs.push_back(0);

    for (label i = 0; i < nLocal; ++i)
    {
        label globalRow = rangeStart + i;

        // Left neighbor
        if (globalRow > 0)
        {
            if (i > 0)
            {
                colIdxs.push_back(static_cast<localIdx>(i - 1));
            }
            else
            {
                colIdxs.push_back(static_cast<localIdx>(leftGhostLocal));
            }
            values.push_back(-1.0 / (h * h));
        }

        // Diagonal
        colIdxs.push_back(static_cast<localIdx>(i));
        values.push_back(2.0 / (h * h));

        // Right neighbor
        if (globalRow < nGlobal - 1)
        {
            if (i < nLocal - 1)
            {
                colIdxs.push_back(static_cast<localIdx>(i + 1));
            }
            else
            {
                colIdxs.push_back(static_cast<localIdx>(rightGhostLocal));
            }
            values.push_back(-1.0 / (h * h));
        }

        rowOffs.push_back(static_cast<localIdx>(colIdxs.size()));
        rhsValues.push_back(1.0);
    }

    // Create NeoN CSR matrix
    auto sparsity = std::make_shared<SparsityPattern<localIdx>>(
        Vector<localIdx>(exec, colIdxs), Vector<localIdx>(exec, rowOffs)
    );
    CSRMatrix<scalar, localIdx> csrMatrix(Vector<scalar>(exec, values), sparsity);
    Vector<scalar> rhs(exec, rhsValues);

    // Convert NeoN local CSR → Ginkgo matrix_data with global indices
    auto hostColIdxs = csrMatrix.colIdxs().copyToHost();
    auto hostRowOffs = csrMatrix.rowOffs().copyToHost();
    auto hostValues = csrMatrix.values().copyToHost();

    gko::matrix_data<scalar, label> matData {
        gko::dim<2> {static_cast<std::size_t>(nGlobal), static_cast<std::size_t>(nGlobal)}
    };
    for (label row = 0; row < nLocal; ++row)
    {
        label globalRow = globalCellIds[static_cast<std::size_t>(row)];
        for (localIdx j = hostRowOffs.data()[row]; j < hostRowOffs.data()[row + 1]; ++j)
        {
            localIdx localCol = hostColIdxs.data()[j];
            label globalCol = (localCol < nLocal)
                                ? globalCellIds[static_cast<std::size_t>(localCol)]
                                : ghostCellGlobalIds[static_cast<std::size_t>(localCol - nLocal)];
            matData.nonzeros.emplace_back(globalRow, globalCol, hostValues.data()[j]);
        }
    }

    gko::matrix_data<scalar, label> rhsGkoData {gko::dim<2> {static_cast<std::size_t>(nGlobal), 1}};
    for (label i = 0; i < nLocal; ++i)
    {
        rhsGkoData.nonzeros.emplace_back(globalCellIds[static_cast<std::size_t>(i)], 0, 1.0);
    }

    // Build Ginkgo distributed types
    auto gkoExec = gko::ReferenceExecutor::create();
    auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);

    using part_type = gko::experimental::distributed::Partition<label, label>;
    auto partition = gko::share(
        part_type::build_from_global_size_uniform(gkoExec, static_cast<label>(nRanks), nGlobal)
    );

    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    auto A = gko::share(dist_mtx::create(gkoExec, comm));
    A->read_distributed(matData, partition);

    auto b = dist_vec::create(gkoExec, comm);
    b->read_distributed(rhsGkoData, partition);

    auto x = dist_vec::create(gkoExec, comm);
    gko::matrix_data<scalar, label> zeroData {gko::dim<2> {static_cast<std::size_t>(nGlobal), 1}};
    x->read_distributed(zeroData, partition);

    // Solve
    using cg = gko::solver::Cg<scalar>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<scalar, label, label>;
    using jacobi = gko::preconditioner::Jacobi<scalar, label>;

    auto solver = cg::build()
                      .with_preconditioner(schwarz::build().with_local_solver(jacobi::build()))
                      .with_criteria(
                          gko::stop::Iteration::build().with_max_iters(200u),
                          gko::stop::ResidualNorm<scalar>::build().with_reduction_factor(1e-10)
                      )
                      .on(gkoExec)
                      ->generate(A);

    solver->apply(b, x);

    // Verify against analytic solution u(x) = x(1-x)/2
    auto localVec = x->get_local_vector();
    for (label i = 0; i < nLocal; ++i)
    {
        scalar xi = static_cast<scalar>(rangeStart + i + 1) * h;
        scalar expected = xi * (1.0 - xi) / 2.0;
        scalar computed = localVec->at(i, 0);
        REQUIRE(computed == Catch::Approx(expected).margin(1e-6));
    }
}

TEST_CASE("Partition metadata - stencilDB entries")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());

    NeoN::Executor exec = NeoN::SerialExecutor();

    // Create a 2D mesh and partition it
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4); // 16 cells
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);

    // Extract sub-mesh for this rank
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    // Verify new stencilDB entries exist
    REQUIRE(subMesh.stencilDB().contains("partition::nGlobalCells"));
    REQUIRE(subMesh.stencilDB().contains("partition::nGhostCells"));
    REQUIRE(subMesh.stencilDB().contains("partition::procBoundaryGhostMap"));
    REQUIRE(subMesh.stencilDB().contains("partition::procBoundaryStartOffset"));

    auto nGlobalCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGlobalCells");
    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");
    auto& ghostMap = *subMesh.stencilDB().get<std::shared_ptr<std::vector<localIdx>>>(
        "partition::procBoundaryGhostMap"
    );
    auto procBndStart =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::procBoundaryStartOffset");

    REQUIRE(nGlobalCells == mesh.nCells());
    REQUIRE(nGhostCells >= 0);

    // Ghost map entries should point to valid ghost cell indices
    for (auto idx : ghostMap)
    {
        REQUIRE(idx >= subMesh.nCells());
        REQUIRE(idx < subMesh.nCells() + nGhostCells);
    }

    // procBoundaryStartOffset should be within boundary mesh range
    REQUIRE(procBndStart >= 0);
    REQUIRE(procBndStart <= subMesh.boundaryMesh().faceCells().size());
}

TEST_CASE("ProcBoundary BC - ghost values propagate")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());

    NeoN::Executor exec = NeoN::SerialExecutor();

    // Create and partition a 2D mesh
    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");

    if (nGhostCells == 0)
    {
        // Single-rank case: no proc boundaries, nothing to test
        return;
    }

    // Get patch names to identify proc-boundary patches
    const auto& patchNames =
        *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");

    // Create BCs: procBoundary for proc patches, fixedValue for physical patches
    std::vector<fvcc::VolumeBoundary<scalar>> bcs;
    for (localIdx patchID = 0; patchID < subMesh.nBoundaries(); patchID++)
    {
        const auto& pName = patchNames[static_cast<std::size_t>(patchID)];
        if (pName.substr(0, 4) == "proc")
        {
            NeoN::Dictionary procDict({{"type", std::string("procBoundary")}});
            bcs.emplace_back(subMesh, procDict, patchID);
        }
        else
        {
            NeoN::Dictionary fvDict(
                {{"type", std::string("fixedValue")}, {"fixedValue", scalar(0.0)}}
            );
            bcs.emplace_back(subMesh, fvDict, patchID);
        }
    }

    // Create a ghost-extended internal vector
    localIdx totalSize = subMesh.nCells() + nGhostCells;
    Vector<scalar> internalVec(exec, totalSize, 0.0);

    // Set ghost cell values to a known value
    auto* data = internalVec.data();
    for (localIdx i = subMesh.nCells(); i < totalSize; ++i)
    {
        data[i] = 42.0;
    }

    // Create VolumeField with ghost-extended vector
    fvcc::VolumeField<scalar> phi(exec, "phi", subMesh, internalVec, bcs);

    // Correct boundary conditions - should read ghost values
    phi.correctBoundaryConditions();

    // Verify that proc-boundary faces have valueFraction=1.0 and value=42.0
    auto valueFractionHost = phi.boundaryData().valueFraction().copyToHost();
    auto refValueHost = phi.boundaryData().refValue().copyToHost();
    auto procBndStart =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::procBoundaryStartOffset");
    auto nBndFaces = subMesh.boundaryMesh().faceCells().size();

    for (localIdx i = procBndStart; i < static_cast<localIdx>(nBndFaces); ++i)
    {
        REQUIRE(valueFractionHost.data()[i] == Catch::Approx(1.0));
        REQUIRE(refValueHost.data()[i] == Catch::Approx(42.0));
    }
}

TEST_CASE("Laplacian - ghost column entries assembled from proc-boundary faces")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());
    NeoN::Executor exec = NeoN::SerialExecutor();

    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");
    if (nGhostCells == 0)
    {
        return;
    }

    // Create phi with ghost-extended vector (ghost cells = 1.0)
    localIdx totalSize = subMesh.nCells() + nGhostCells;
    NeoN::Vector<scalar> internalVec(exec, totalSize, 1.0);
    auto bcs = makePartitionedBCs(subMesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", subMesh, internalVec, bcs);
    phi.correctBoundaryConditions();

    // Create gamma = 1 on all faces
    auto surfBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(subMesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", subMesh, surfBCs);
    fill(gamma.internalVector(), scalar(1.0));
    fill(gamma.boundaryData().value(), scalar(1.0));

    // Build LinearSystem from partitioned sparsity (Phase 1 adds ghost columns)
    auto sparsity = NeoN::la::createSparsityPatternFaceToMatrixAddress<localIdx>(subMesh);
    NeoN::la::LinearSystem<scalar> ls(sparsity);

    // Assemble laplacian implicitly
    NeoN::Input input =
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});
    fvcc::LaplacianOperator<scalar> lapOp(NeoN::dsl::Operator::Type::Implicit, gamma, phi, input);
    lapOp.implicitOperation(ls);

    // Ghost column entries must be non-zero after assembly
    auto colIdxsH = sparsity->sparsityPattern()->colIdxs().copyToHost();
    auto valuesH = ls.matrix().values().copyToHost();
    localIdx nCells = subMesh.nCells();

    bool hasNonZeroGhostEntry = false;
    for (localIdx i = 0; i < static_cast<localIdx>(colIdxsH.size()); ++i)
    {
        if (colIdxsH.data()[i] >= nCells && valuesH.data()[i] != scalar(0.0))
        {
            hasNonZeroGhostEntry = true;
            break;
        }
    }
    REQUIRE(hasNonZeroGhostEntry);
}

TEST_CASE("FaceToMatrixAddress - ghost columns from partitioned mesh")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());
    NeoN::Executor exec = NeoN::SerialExecutor();

    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");
    if (nGhostCells == 0)
    {
        return; // single-rank: no proc boundaries
    }

    auto mi = NeoN::la::createSparsityPatternFaceToMatrixAddress<localIdx>(subMesh);
    localIdx nCells = subMesh.nCells();

    // sparsity must contain at least one ghost column entry (colIdx >= nCells)
    auto colIdxsH = mi->sparsityPattern()->colIdxs().copyToHost();
    bool hasGhostColumn = false;
    for (localIdx i = 0; i < static_cast<localIdx>(colIdxsH.size()); ++i)
    {
        if (colIdxsH.data()[i] >= nCells)
        {
            hasGhostColumn = true;
            break;
        }
    }
    REQUIRE(hasGhostColumn);

    // procBoundaryOffset accessor must exist and be non-empty
    REQUIRE(mi->procBoundaryOffset().size() > 0);
}

TEST_CASE("Communicator on mesh - isDistributed")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());

    NeoN::Executor exec = NeoN::SerialExecutor();

    auto mesh = NeoN::createUniform2DGrid(exec, 4, 4);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    // Initially not distributed
    REQUIRE(!subMesh.isDistributed());

    // Build and set communicator
    auto& nPartsPtr = subMesh.stencilDB().get<std::shared_ptr<int>>("partition::nParts");
    auto& sendData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commSendMap"
    );
    auto& recvData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commReceiveMap"
    );

    NeoN::CommMap sendMap(mpiEnv.sizeRank()), receiveMap(mpiEnv.sizeRank());
    for (int r = 0; r < *nPartsPtr && r < static_cast<int>(mpiEnv.sizeRank()); ++r)
    {
        for (auto idx : sendData[static_cast<std::size_t>(r)])
            sendMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
        for (auto idx : recvData[static_cast<std::size_t>(r)])
            receiveMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
    }

    subMesh.setCommunicator(NeoN::Communicator(mpiEnv, sendMap, receiveMap));
    REQUIRE(subMesh.isDistributed());
}

TEST_CASE("toGlobalMatrixData - local CSR with ghost column converts to global indices")
{
    // 2 local cells (global IDs 10, 11), 1 ghost cell (global ID 5)
    // Matrix:
    //   row 10: [diag=2.0 at col 10, off=-1.0 at col 11]
    //   row 11: [off=-1.0 at col 10, diag=2.0 at col 11, ghost=-1.0 at col 5]
    NeoN::Executor exec = NeoN::SerialExecutor();

    std::vector<localIdx> colIdxVec = {0, 1, 0, 1, 2};
    std::vector<localIdx> rowOffsVec = {0, 2, 5};
    std::vector<scalar> valuesVec = {2.0, -1.0, -1.0, 2.0, -1.0};

    using NeoN::la::SparsityPattern;
    using NeoN::la::CSRMatrix;
    auto sp = std::make_shared<SparsityPattern<localIdx>>(
        Vector<localIdx>(exec, colIdxVec), Vector<localIdx>(exec, rowOffsVec)
    );
    CSRMatrix<scalar, localIdx> mat(Vector<scalar>(exec, valuesVec), sp);

    std::vector<label> globalCellIds = {10, 11};
    std::vector<label> ghostCellGlobalIds = {5};
    localIdx nLocalCells = 2;

    // Call the utility (will fail: function doesn't exist yet)
    auto matData =
        NeoN::la::toGlobalMatrixData(mat, globalCellIds, ghostCellGlobalIds, nLocalCells);

    // Verify global indices
    REQUIRE(matData.nonzeros.size() == 5);
    // Sort for deterministic checks
    matData.sort_row_major();
    // row 10: (10,10,2.0) (10,11,-1.0)
    REQUIRE(matData.nonzeros[0].row == 10);
    REQUIRE(matData.nonzeros[0].column == 10);
    REQUIRE(matData.nonzeros[0].value == Catch::Approx(2.0));
    REQUIRE(matData.nonzeros[1].row == 10);
    REQUIRE(matData.nonzeros[1].column == 11);
    REQUIRE(matData.nonzeros[1].value == Catch::Approx(-1.0));
    // row 11: (11,5,-1.0) (11,10,-1.0) (11,11,2.0)  [sorted by col]
    REQUIRE(matData.nonzeros[2].row == 11);
    REQUIRE(matData.nonzeros[2].column == 5);
    REQUIRE(matData.nonzeros[2].value == Catch::Approx(-1.0));
    REQUIRE(matData.nonzeros[3].row == 11);
    REQUIRE(matData.nonzeros[3].column == 10);
    REQUIRE(matData.nonzeros[3].value == Catch::Approx(-1.0));
    REQUIRE(matData.nonzeros[4].row == 11);
    REQUIRE(matData.nonzeros[4].column == 11);
    REQUIRE(matData.nonzeros[4].value == Catch::Approx(2.0));
}

TEST_CASE("DistributedGinkgoSolver - partitioned 2D Laplacian")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());
    NeoN::Executor exec = NeoN::SerialExecutor();

    auto mesh = NeoN::createUniform2DGrid(exec, 6, 6);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");

    // Build communicator so isDistributed() returns true
    auto& nPartsPtr = subMesh.stencilDB().get<std::shared_ptr<int>>("partition::nParts");
    auto& sendData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commSendMap"
    );
    auto& recvData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commReceiveMap"
    );
    NeoN::CommMap sendMap(mpiEnv.sizeRank()), receiveMap(mpiEnv.sizeRank());
    for (int r = 0; r < *nPartsPtr && r < static_cast<int>(mpiEnv.sizeRank()); ++r)
    {
        for (auto idx : sendData[static_cast<std::size_t>(r)])
            sendMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
        for (auto idx : recvData[static_cast<std::size_t>(r)])
            receiveMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
    }
    subMesh.setCommunicator(NeoN::Communicator(mpiEnv, sendMap, receiveMap));

    // Create phi with ghost-extended vector (all zeros), fixedValue=1 on boundaries
    localIdx totalSize = subMesh.nCells() + nGhostCells;
    NeoN::Vector<scalar> internalVec(exec, totalSize, 0.0);
    auto bcs = makePartitionedBCs(subMesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", subMesh, internalVec, bcs);
    phi.correctBoundaryConditions();

    auto surfBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(subMesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", subMesh, surfBCs);
    fill(gamma.internalVector(), scalar(1.0));
    fill(gamma.boundaryData().value(), scalar(1.0));

    auto sparsity = NeoN::la::createSparsityPatternFaceToMatrixAddress<localIdx>(subMesh);
    NeoN::la::LinearSystem<scalar> ls(sparsity);

    NeoN::Input input =
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});
    fvcc::LaplacianOperator<scalar> lapOp(NeoN::dsl::Operator::Type::Implicit, gamma, phi, input);
    lapOp.implicitOperation(ls);

    // Solve with DistributedGinkgoSolver (will fail: class doesn't exist yet)
    NeoN::Dictionary solverCfg(
        {{"solver", std::string("CG")}, {"preconditioner", std::string("Jacobi")}}
    );
    NeoN::la::DistributedGinkgoSolver solver(exec, solverCfg, subMesh);
    auto stats = solver.solve(ls, phi.internalVector());

    // Solution should have converged
    REQUIRE(stats.entries.back().finalResNorm < 1e-6);
}

TEST_CASE("dsl::solve - distributed Laplacian via DistributedGinkgoSolver")
{
    NeoN::mpi::MPIEnvironment mpiEnv;
    auto rank = static_cast<int>(mpiEnv.rank());
    auto nRanks = static_cast<int>(mpiEnv.sizeRank());
    NeoN::Executor exec = NeoN::SerialExecutor();

    auto mesh = NeoN::createUniform2DGrid(exec, 6, 6);
    auto cellPart = NeoN::partition::partitionMesh(mesh, nRanks);
    auto subMesh = NeoN::partition::extractSubMesh(mesh, cellPart, rank);

    auto nGhostCells =
        *subMesh.stencilDB().get<std::shared_ptr<localIdx>>("partition::nGhostCells");

    // Build communicator
    auto& nPartsPtr = subMesh.stencilDB().get<std::shared_ptr<int>>("partition::nParts");
    auto& sendData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commSendMap"
    );
    auto& recvData = *subMesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commReceiveMap"
    );
    NeoN::CommMap sendMap(mpiEnv.sizeRank()), receiveMap(mpiEnv.sizeRank());
    for (int r = 0; r < *nPartsPtr && r < static_cast<int>(mpiEnv.sizeRank()); ++r)
    {
        for (auto idx : sendData[static_cast<std::size_t>(r)])
            sendMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
        for (auto idx : recvData[static_cast<std::size_t>(r)])
            receiveMap[static_cast<std::size_t>(r)].push_back(
                NeoN::NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
    }
    subMesh.setCommunicator(NeoN::Communicator(mpiEnv, sendMap, receiveMap));

    localIdx totalSize = subMesh.nCells() + nGhostCells;
    NeoN::Vector<scalar> internalVec(exec, totalSize, 0.0);
    auto bcs = makePartitionedBCs(subMesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", subMesh, internalVec, bcs);
    phi.correctBoundaryConditions();

    auto surfBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(subMesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", subMesh, surfBCs);
    fill(gamma.internalVector(), scalar(1.0));
    fill(gamma.boundaryData().value(), scalar(1.0));

    // Build DSL expression: implicit laplacian
    NeoN::Input lapInput =
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")});
    NeoN::dsl::SpatialOperator lapOp = NeoN::dsl::imp::laplacian(gamma, phi);
    lapOp.read(lapInput);
    NeoN::dsl::Expression<scalar> eqn(lapOp);

    // fvSchemes with dummy timeIntegration (required by dsl::solve even for pure spatial)
    NeoN::Dictionary timeIntDict;
    timeIntDict.insert("type", std::string("backwardEuler"));
    NeoN::Dictionary laplacianSchemesDict;
    laplacianSchemesDict.insert(
        "laplacian(gamma,phi)",
        NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")})
    );
    NeoN::Dictionary fvSchemes;
    fvSchemes.insert("timeIntegration", timeIntDict);
    fvSchemes.insert("laplacianSchemes", laplacianSchemesDict);

    // fvSolution selects DistributedGinkgoSolver via the new distributed path
    NeoN::Dictionary fvSolution;
    fvSolution.insert("solver", std::string("DistributedGinkgo"));

    // Will fail (key not found) until DistributedGinkgo is registered/wired in dsl::solve
    auto stats = NeoN::dsl::solve(eqn, phi, scalar(0), scalar(1), fvSchemes, fvSolution);

    REQUIRE(stats.entries.back().finalResNorm < 1e-6);
}
