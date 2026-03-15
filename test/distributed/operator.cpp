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
auto partitionMeshHelper(auto mesh, NeoN::mpi::Environment mpiEnviron)
{

    // rank 0 takes first 1/3 cells

    // original
    // [ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 ]
    // partitioned
    //   0   1   2   3     4    5   6   7     8    9
    // [ 0 | 1 | 2 | 3 ] [ 0  | 1 | 2 | 3 ] [  | | |]
    //     0   1   2    3     4   5   6    7  8 9 10             // global faces

    auto exec = mesh.exec();

    localIdx localCells = mesh.nCells() / mpiEnviron.sizeRank(); // 4
    localIdx localFaces = localCells - 1;
    localIdx rankOffset = 0;

    localIdx firstCell = 0;
    localIdx lastCell = mesh.nCells();

    localIdx firstFace = 0;
    localIdx lastFace = mesh.nInternalFaces() - 1;

    if (mpiEnviron.rank() == 0)
    {
        lastCell = localCells;
        lastFace = localFaces;
    }

    if (mpiEnviron.rank() == 1)
    {
        firstCell = localCells;
        firstFace = localFaces + 1; // skip one face since it is a boundary now
        lastCell = localCells + localCells;
        lastFace = firstFace + localFaces;
        rankOffset = localCells;
    }

    if (mpiEnviron.rank() == 2)
    {
        firstCell = localCells + localCells;
        firstFace = firstCell;
        rankOffset = 2 * localCells;
    }

    auto points = take(mesh.points(), firstCell, lastCell);
    auto cellVolumes = take(mesh.cellVolumes(), firstCell, lastCell);
    auto cellCentres = take(mesh.cellCentres(), firstCell, lastCell);

    auto faceAreas = take(mesh.faceAreas(), firstFace, lastFace);
    auto faceCentres =
        take(mesh.faceCentres(), firstFace, lastFace); // this includes also boundary faces
    auto magFaceAreas = take(mesh.magFaceAreas(), firstFace, lastFace);

    // FIXME need to subtract offset
    auto faceOwner = take(mesh.faceOwner(), firstFace, lastFace);
    auto faceNeighbour = take(mesh.faceNeighbour(), firstFace, lastFace);
    sub(faceOwner, rankOffset);
    sub(faceNeighbour, rankOffset);

    auto nCells = localCells;
    auto nInternalFaces = localFaces;
    auto nBoundaryFaces = mesh.nBoundaryFaces(); // stays constant at 2
    auto nBoundaries = mesh.nBoundaries();       // stays constant
    auto nFaces = nInternalFaces + nBoundaryFaces;

    auto cellCentresPart = take(mesh.cellCentres(), firstCell, lastCell);

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, localCells - 1}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}}, // FIXME
        cellCentresPart,
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        {exec, {1.0, 1.0}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        mesh.boundaryMesh().delta(), // FIXME
        {exec, {1.0, 1.0}},
        mesh.boundaryMesh().deltaCoeffs(),
        {0, 1, 2},
        {-1, -1, -1}
    );

    NF_PING();

    return UnstructuredMesh(
        exec,
        points,
        cellVolumes,
        cellCentres,
        faceAreas,
        faceCentres,
        magFaceAreas,
        faceOwner,
        faceNeighbour,
        localCells,
        nInternalFaces,
        nBoundaryFaces,
        nBoundaries,
        localFaces,
        boundaryMesh
    );
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
FieldType partitionVolField(FieldType field, auto mesh, auto bcs, size_t rank)
{


    return {field.exec(), field.name + "Part", mesh, bcs};
}

/** @brief helper function given a 1D uniform mesh and a rank it will return the part of the mesh
 owned by this rank */
template<typename FieldType>
FieldType partitionSurfaceField(FieldType field, auto mesh, auto bcs, size_t rank)
{


    return {field.exec(), field.name + "Part", mesh, bcs};
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
        {"divSchemes",
         NeoN::Dictionary {
             {"div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
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
        {"divSchemes",
         NeoN::Dictionary {
             {"div(phiPart,UPart)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
         }}
    };

    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NF_PING();

    auto nCells = 12;
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto mesh = create1DUniformMesh(exec, nCells);
    NF_PING();

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );
    NF_PING();

    // randomizeVector(U);

    // auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    // auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    // auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh,
    // surfaceBCs);

    NF_PING();
    // fill(phi.internalVector(), 1.0);
    // randomizeVector(phi.internalVector());
    NF_PING();
    // fill(gamma.internalVector(), 2.0);

    // partition fields and data

    // assembly
    // auto expr = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phi, U)
    // ); // - NeoN::dsl::imp::laplacian(gamma, U);
    // expr.read(input);
    // auto [sp, ls] = expr.assemble(mesh, 1.0, 1.0);

    // SECTION("Can assemble distributed " + execName)
    // {
    NF_PING();
    NeoN::mpi::Environment mpiEnviron;

    NF_PING();
    auto meshPart = partitionMeshHelper(meshGlobal, mpiEnviron);
    NF_PING();
    auto volBCsII = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshPart);
    NF_PING();
    auto volBCsPart = setProcessorBoundaryHelper(volBCsII, mpiEnviron.rank());
    NF_PING();
    auto uPart = partitionVolField(U, meshPart, volBCsPart, mpiEnviron.rank());
    NF_PING();
    auto surfaceBCsII = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(meshPart);
    NF_PING();
    auto surfaceBCsPart = setProcessorBoundaryHelper(surfaceBCsII, mpiEnviron.rank());
    // auto phiPart = partitionSurfaceField(phi, meshPart, surfaceBCsPart, mpiEnviron.rank());
    auto phiPart =
        finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phiPart", meshPart, surfaceBCsPart);
    NF_PING();

    auto exprDist = NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(phiPart, uPart)
    ); // - NeoN::dsl::imp::laplacian(gamma, U);

    try
    {
        exprDist.read(inputPart);
    }
    catch (const std::bad_variant_access& e)
    {
        std::cout << __FILE__ << " : " << __LINE__ << " caught exception\n";
        std::cout << e.what() << '\n';
    }

    mpi::Environment env;

    std::vector<localIdx> commIdx {};
    std::vector<int> sendCounts {};
    std::vector<int> commRanks {};
    std::vector<localIdx> boundaryMapVector {};

    size_t boundaryMapSize = 1;

    if (env.rank() == 0)
    {
        // communicate the interior value which is
        commIdx = std::vector<localIdx> {1};
        sendCounts = std::vector<int> {0, 1, 0, 1};
        boundaryMapVector = std::vector<localIdx> {13};
    }
    if (env.rank() == 1)
    {
        // communicate the interior value which is
        boundaryMapSize = 1;
        commIdx = std::vector<localIdx> {0, 1};
        sendCounts = std::vector<int> {1, 0, 1, 2};
        boundaryMapVector = std::vector<localIdx> {0, 13};
    }
    if (env.rank() == 2)
    {
        // communicate the interior value which is
        commIdx = std::vector<localIdx> {0};
        sendCounts = std::vector<int> {0, 1, 0, 1};
        boundaryMapVector = std::vector<localIdx> {0};
    }

    // map from proc boundary to matrix values address
    Vector<localIdx> boundaryMatrixMap {exec, boundaryMapVector};

    auto commPattern = CommunicationPattern(commIdx, sendCounts, env);
    // auto [spDst, lsDst] =
    //     exprDist.assembleDistributed(meshPart, 1.0, 1.0, commPattern, boundaryMatrixMap);

    // auto lsDstH = lsDst.matrix().values().copyToHost();

    // // matrix values
    // if (env.rank() == 0)
    // {
    //     compare(ls.matrix().diag(), lsDst.matrix().diag(), ApproxScalar(1e-15));
    //     compare(ls.matrix().values(), lsDst.matrix().values(), ApproxScalar(1e-15));
    // }

    // auto matDiag = ls.matrix().diag();
    // auto matOptDiag = lsOpt.matrix().diag();
    // SECTION("Has correct diagonal") { compare(matDiag, matOptDiag, ApproxScalar(epsilon)); }

    // auto matUpper = upper(ls.matrix());
    // auto matOptUpper = upper(lsOpt.matrix());
    // SECTION("Has correct upper") { compare(matUpper, matOptUpper, ApproxScalar(epsilon)); }
    // }
}

}
