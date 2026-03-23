// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
auto partitionMeshHelper(auto& mesh, NeoN::mpi::Environment mpiEnviron)
{
    auto exec = mesh.exec();
    localIdx localCells = mesh.nCells() / mpiEnviron.sizeRank(); // 4
    auto ret = create1DUniformMesh(exec, localCells);

    if (mpiEnviron.rank() != 0)
    {
        // FIXME NOTE -1.0 should be 1.0 ?
        ret.boundaryMesh().nf().view()[0] = {-1.0, 0.0, 0.0};
        ret.boundaryMesh().sf().view()[0] = {-1.0, 0.0, 0.0};
    }

    return ret;
}

/** @brief helper function to set the processor boundaries of a distributed field */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(std::vector<BoundaryType> bcs, size_t rank)
{
    return bcs;
}

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType
partitionVolField(FieldType field, auto& mesh, auto bcs, NeoN::mpi::Environment mpiEnviron)
{
    localIdx localCells = mesh.nCells();
    localIdx firstCell = 0;
    localIdx lastCell = localCells;

    if (mpiEnviron.rank() == 0)
    {
        lastCell = localCells;
    }
    if (mpiEnviron.rank() == 1)
    {
        firstCell = localCells;
        lastCell = localCells + localCells;
    }
    if (mpiEnviron.rank() == 2)
    {
        firstCell = localCells + localCells;
        lastCell = localCells + localCells + localCells;
    }

    auto internalVector = take(field.internalVector(), firstCell, lastCell);

    return {field.exec(), field.name + "Part", mesh, internalVector, bcs};
}

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType
partitionSurfaceField(FieldType field, auto& mesh, auto bcs, NeoN::mpi::Environment mpiEnviron)
{
    auto exec = mesh.exec();
    localIdx localCells = mesh.nCells();  // 4
    localIdx localFaces = localCells - 1; // 3
    localIdx firstFace = 0;
    localIdx lastFace = localFaces;

    localIdx leftBoundaryFace = 0;
    localIdx rightBoundaryFace = 0;


    // [ 0 | 1 | 2 | 3 ][ 4 | 5 | 6 | 7 ][ 8 | 9 | 10 | 11 ]
    // 11  0   1   2   3    4   5   6   7    8   9   10   12

    if (mpiEnviron.rank() == 0)
    {
        lastFace = localFaces + 1;             // 4
        leftBoundaryFace = 3 * localFaces + 2; // 11
        rightBoundaryFace = localFaces;        // 3
    }
    if (mpiEnviron.rank() == 1)
    {
        firstFace = localFaces + 1;            // 4  last face rank 0
        lastFace = firstFace + localFaces + 1; // 8

        leftBoundaryFace = localFaces;                     // should be 3
        rightBoundaryFace = leftBoundaryFace + localCells; // should 3 + 4
    }
    if (mpiEnviron.rank() == 2)
    {
        firstFace = localCells + localCells; // 8 last face rank 1
        lastFace = firstFace + localFaces + 1;

        leftBoundaryFace = 2 * localFaces + 1;                 // 7
        rightBoundaryFace = leftBoundaryFace + localCells + 1; // 12
    }

    FieldType ret = {field.exec(), field.name + "Part", mesh, bcs};

    // NOTE last two values are boundaries and are overwritten next
    auto internalVector = take(field.internalVector(), firstFace, lastFace + 1);
    // value lastFace  and lastFace+1 are incorrect
    // value lastFace is the left boundary
    // and lastFace + 1 is at the new boundary so it should be lastFace

    auto outV = internalVector.view();
    auto inV = field.internalVector().view();

    // set left boundary face
    NeoN::parallelFor(
        // lastface
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) { outV[localFaces] = inV[leftBoundaryFace]; },
        "copyMap"
    );

    NeoN::parallelFor(
        exec,
        {0, 1},
        NEON_LAMBDA(const localIdx i) { outV[localFaces + 1] = inV[rightBoundaryFace]; },
        "copyMap"
    );

    NF_ASSERT(ret.internalVector().size() == internalVector.size(), "different size");
    ret.internalVector() = internalVector;
    return ret;
}

TEST_CASE("Distributed")
{
    // start with non distributed setup
    float epsilon = 1e-32;

    auto input = NeoN::Dictionary {
        {
            "laplacianSchemes",
            NeoN::Dictionary {
                {"laplacian(gamma,U)",
                 NeoN::TokenList(
                     {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                 )}
            },
        },
        // FIXME use upwind again
        {"divSchemes",
         NeoN::Dictionary {
             {"div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})}
         }}
    };

    auto inputPart = NeoN::Dictionary {
        {
            "laplacianSchemes",
            NeoN::Dictionary {
                {"laplacian(gamma,UPart)",
                 NeoN::TokenList(
                     {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                 )}
            },
        },
        // FIXME use upwind again
        {"divSchemes",
         NeoN::Dictionary {
             {"div(phiPart,UPart)", NeoN::TokenList({std::string("Gauss"), std::string("linear")})}
         }}
    };

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 12;
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    // randomizeVector(U);

    // auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    // auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    // auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh,
    // surfaceBCs);

    // fill(phi.internalVector(), 1.0);
    // randomizeVector(phi.internalVector());
    // fill(gamma.internalVector(), 2.0);

    // // partition fields and data

    // // assembly
    // auto expr = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phi, U)
    // ); // - NeoN::dsl::imp::laplacian(gamma, U);
    // expr.read(input);
    // auto [sp, ls] = expr.assemble(mesh, 1.0, 1.0);

    // // SECTION("Can assemble distributed " + execName)
    // // {
    // NeoN::mpi::Environment mpiEnviron;

    // auto meshPart = partitionMeshHelper(meshGlobal, mpiEnviron);
    // auto volBCsII = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshPart);
    // auto volBCsPart = setProcessorBoundaryHelper(volBCsII, mpiEnviron.rank());
    // auto uPart = partitionVolField(U, meshPart, volBCsPart, mpiEnviron);
    // auto surfaceBCsII = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(meshPart);
    // auto surfaceBCsPart = setProcessorBoundaryHelper(surfaceBCsII, mpiEnviron.rank());
    // auto phiPart = partitionSurfaceField(phi, meshPart, surfaceBCsPart, mpiEnviron);

    // auto exprDist = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phiPart, uPart)
    // ); // - NeoN::dsl::imp::laplacian(gamma, U);

    // exprDist.read(inputPart);

    // mpi::Environment env;

    // std::vector<localIdx> commIdx {};
    // std::vector<int> sendCounts {};
    // std::vector<int> commRanks {};
    // std::vector<localIdx> boundaryMapVector {};

    // size_t boundaryMapSize = 1;

    // if (env.rank() == 0)
    // {
    //     // communicate the interior value which is
    //     commIdx = std::vector<localIdx> {1};
    //     sendCounts = std::vector<int> {0, 1, 0, 1};
    //     boundaryMapVector = std::vector<localIdx> {9};
    // }
    // if (env.rank() == 1)
    // {
    //     // communicate the interior value which is
    //     boundaryMapSize = 1;
    //     commIdx = std::vector<localIdx> {0, 1};
    //     sendCounts = std::vector<int> {1, 0, 1, 2};
    //     boundaryMapVector = std::vector<localIdx> {0, 9};
    // }
    // if (env.rank() == 2)
    // {
    //     // communicate the interior value which is
    //     commIdx = std::vector<localIdx> {0};
    //     sendCounts = std::vector<int> {0, 1, 0, 1};
    //     boundaryMapVector = std::vector<localIdx> {0};
    // }

    // // map from proc boundary to matrix values address
    // Vector<localIdx> boundaryMatrixMap {exec, boundaryMapVector};

    // auto commPattern = CommunicationPattern(commIdx, sendCounts, env);
    // auto [spDst, lsDst] =
    //     exprDist.assembleDistributed(meshPart, 1.0, 1.0, commPattern, boundaryMatrixMap);

    // localIdx firstElement = 0;
    // localIdx lastElement = 0;

    // if (mpiEnviron.rank() == 0)
    // {
    //     lastElement = 10;
    // }
    // if (mpiEnviron.rank() == 1)
    // {
    //     firstElement = 12;
    //     lastElement = 22;
    // }
    // if (mpiEnviron.rank() == 2)
    // {
    //     firstElement = 24;
    //     lastElement = 34;
    // }

    // if (env.rank() == 0)
    // {
    //     compare(
    //         take(ls.matrix().values(), firstElement, lastElement),
    //         lsDst.matrix().values(),
    //         ApproxScalar(1e-15)
    //     );
    // }
    // // if (env.rank() == 1)
    // // {
    // //     compare(
    // //         take(ls.matrix().values(), firstElement, lastElement),
    // //         lsDst.matrix().values(),
    // //         ApproxScalar(1e-15)
    // //     );
    // // }
    // // if (env.rank() == 2)
    // // {
    // //     compare(
    // //         take(ls.matrix().values(), firstElement, lastElement),
    // //         lsDst.matrix().values(),
    // //         ApproxScalar(1e-15)
    // //     );
    // // }

    // Dictionary solverDict {
    //     {{"solver", std::string {"Ginkgo"}},
    //      {"type", "solver::Cg"},
    //      {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
    // };

    // // Create solver
    // auto solver = NeoN::la::Solver(exec, solverDict);
    // auto x = Vector<scalar>(exec, 4);
    // fill(x, 0.0);

    // auto solverStats = solver.solve(lsDst, x);
    // auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];
}

}
