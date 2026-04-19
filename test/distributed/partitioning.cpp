// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

/** @brief helper function to set the processor boundaries of a distributed field */
template<typename BoundaryType>
auto setProcessorBoundaryHelper(
    UnstructuredMesh& mesh, const std::vector<BoundaryType>& bcs, size_t rank
)
{
    auto ret = std::vector<BoundaryType> {};
    for (int i = 0; i < bcs.size(); i++)
    {
        std::string boundaryType = "calculated";
        if (rank == 0 && i == 1)
        {
            boundaryType = "processor";
        }
        if (rank == 2 && i == 0)
        {
            boundaryType = "processor";
        }
        if (rank == 1)
        {
            boundaryType = "processor";
        }
        auto patchDict = Dictionary({{"type", boundaryType}});
        ret.emplace_back(mesh, patchDict, i);
    }
    return ret;
}

TEST_CASE("Distributed")
{
    // start with non distributed setup
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // global number of cells
    auto nCells = 12;
    NeoN::mpi::Environment mpiEnviron;
    auto meshPart = create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank());

    SECTION("Has correct partitioned 1d mesh" + execName)
    {
        REQUIRE(meshPart.nCells() == nCells / mpiEnviron.sizeRank());
        REQUIRE(meshPart.nInternalFaces() == 3);
        REQUIRE(meshPart.boundaryMesh().isDistributed());

        SECTION_IF(mpiEnviron.rank() == 0, "Rank == 0 has correct proc boundary " + execName)
        {
            REQUIRE(meshPart.nBoundaryFaces() == 1);
            REQUIRE(meshPart.nProcBoundaryFaces() == 1);
            REQUIRE(meshPart.boundaryMesh().nProcBoundaryPatches() == 1);
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            REQUIRE(meshPart.nBoundaryFaces() == 0);
            REQUIRE(meshPart.nProcBoundaryFaces() == 2);
            REQUIRE(meshPart.boundaryMesh().nProcBoundaryPatches() == 2);
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank == 2 has correct proc boundary " + execName)
        {
            REQUIRE(meshPart.nBoundaryFaces() == 1);
            REQUIRE(meshPart.nProcBoundaryFaces() == 1);
            REQUIRE(meshPart.boundaryMesh().nProcBoundaryPatches() == 1);
        }
    }

    SECTION("Can create correct communication pattern " + execName)
    {
        auto commPattern = computeCommunicationPattern(meshPart);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            auto sendCountsExp = std::vector<int> {0, 1, 0, 1};
            auto recvIdxExp = std::vector<int> {4};
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            auto sendCountsExp = std::vector<int> {1, 0, 1, 2};
            auto recvIdxExp = std::vector<int> {3, 8};
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto sendCountsExp = std::vector<int> {0, 1, 0, 1};
            auto recvIdxExp = std::vector<int> {7};
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
    }

    auto mesh = create1DUniformMesh(exec, nCells);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec,
        "U",
        mesh,
        Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}),
        volBCs
    );

    auto volBCsII = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshPart);
    auto volBCsPart = setProcessorBoundaryHelper(meshPart, volBCsII, mpiEnviron.rank());
    auto uPart = partitionVolField(U, meshPart, volBCsPart, mpiEnviron);
    uPart.correctBoundaryConditions();

    SECTION("Has correct partitioned VolumeField" + execName)
    {
        REQUIRE(uPart.internalVector().size() == nCells / mpiEnviron.sizeRank());
        REQUIRE(uPart.boundaryData().nBoundaries() == 2);
        // TODO add other ranks
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct " + execName)
        {
            auto uPartExp = std::vector<scalar> {1.0, 2.0, 3.0, 4.0};
            REQUIRE_THAT(uPartExp, IsEqualTo(uPart.internalVector()));
            auto uPartBoundExp = std::vector<scalar> {0.0, 5.0};
            REQUIRE_THAT(uPartBoundExp, IsEqualTo(uPart.boundaryData().value()));
        }
    }

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto phiInternal =
        Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 20.0, 30.0});
    phi.internalVector() = phiInternal;

    auto surfaceBCsII = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(meshPart);
    auto surfaceBCsPart = setProcessorBoundaryHelper(meshPart, surfaceBCsII, mpiEnviron.rank());
    auto phiPart = partitionSurfaceField(phi, meshPart, surfaceBCsPart, mpiEnviron);
    SECTION("Has correct partitioned SurfaceField" + execName)
    {
        REQUIRE(phiPart.boundaryData().nBoundaries() == 2);
        // NOTE
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {1.0, 2.0, 3.0, 20.0, 4.0};
            REQUIRE_THAT(phiExp, IsEqualTo(phiPart.internalVector()));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {5.0, 6.0, 7.0, 4.0, 8.0};
            REQUIRE_THAT(phiExp, IsEqualTo(phiPart.internalVector()));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {9.0, 10.0, 11.0, 30.0, 8.0};
            REQUIRE_THAT(phiExp, IsEqualTo(phiPart.internalVector()));
        }
    }

    SECTION("Can produce correct nonLocalSparsityPattern" + execName)
    {
        auto [mi, commPattern] =
            NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(meshPart);
        auto sp = mi->sparsityPattern();

        SECTION("Can produce internal rowOffs and colIdx " + execName)
        {
            auto rowPtrExp = std::vector<localIdx> {0, 2, 5, 8, 10};
            auto colIdxExp = std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

            REQUIRE_THAT(rowPtrExp, IsEqualTo(sp->rowOffs(), EqualInt()));
            REQUIRE_THAT(colIdxExp, IsEqualTo(sp->colIdxs(), EqualInt()));
        }

        auto bsp = mi->boundarySparsityPattern();
        auto nsp = mi->nonLocalSparsityPattern();
        auto rowToDiagonalMap = la::computeRowToDiagonalMap(nsp->rowOffs(), mi);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            REQUIRE_THAT(I({0}), IsEqualTo(bsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({3}), IsEqualTo(nsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({4}), IsEqualTo(nsp->colIdxs(), EqualInt()));
            REQUIRE_THAT(I({9}), IsEqualTo(rowToDiagonalMap, EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            REQUIRE_THAT(std::vector<localIdx> {}, IsEqualTo(bsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({0, 3}), IsEqualTo(nsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({3, 8}), IsEqualTo(nsp->colIdxs(), EqualInt()));
            REQUIRE_THAT(I({0, 9}), IsEqualTo(rowToDiagonalMap, EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 0 has correct proc boundary " + execName)
        {
            REQUIRE_THAT(I({3}), IsEqualTo(bsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({0}), IsEqualTo(nsp->rowOffs(), EqualInt()));
            REQUIRE_THAT(I({7}), IsEqualTo(nsp->colIdxs(), EqualInt()));
            REQUIRE_THAT(I({0}), IsEqualTo(rowToDiagonalMap, EqualInt()));
        }
    }
}

}
