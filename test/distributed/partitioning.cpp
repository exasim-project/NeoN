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
auto setProcessorBoundaryHelper(std::vector<BoundaryType> bcs, size_t rank)
{
    return bcs;
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
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    auto volBCsII = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshPart);
    auto volBCsPart = setProcessorBoundaryHelper(volBCsII, mpiEnviron.rank());
    auto uPart = partitionVolField(U, meshPart, volBCsPart, mpiEnviron);

    SECTION("Has correct partitioned VolumeField" + execName)
    {
        REQUIRE(uPart.internalVector().size() == nCells / mpiEnviron.sizeRank());
        REQUIRE(uPart.boundaryData().nBoundaries() == 2);
    }

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto phiInternal =
        Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 20.0, 30.0});
    phi.internalVector() = phiInternal;

    auto surfaceBCsII = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(meshPart);
    auto surfaceBCsPart = setProcessorBoundaryHelper(surfaceBCsII, mpiEnviron.rank());
    auto phiPart = partitionSurfaceField(phi, meshPart, surfaceBCsPart, mpiEnviron);
    SECTION("Has correct partitioned SurfaceField" + execName)
    {
        REQUIRE(phiPart.boundaryData().nBoundaries() == 2);
        // NOTE
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {1.0, 2.0, 3.0, 20.0, 4.2};
            // REQUIRE_THAT(phiPart.internalVector(), EqualsRange(phiExp));
            REQUIRE_THAT(phiExp, IsEqualTo(phiPart.internalVector()));
            // compare(phiPart.internalVector(), phiExp, ApproxScalar(1e-15));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            auto phiExp = Vector<scalar>(exec, {5.0, 6.0, 7.0, 4.0, 8.0});
            compare(phiPart.internalVector(), phiExp, ApproxScalar(1e-15));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto phiExp = Vector<scalar>(exec, {9.0, 10.0, 11.0, 30.0, 8.0});
            compare(phiPart.internalVector(), phiExp, ApproxScalar(1e-15));
        }
    }

    SECTION("Can produce correct nonLocalSparsityPattern" + execName)
    {
        auto [mi, commPattern] =
            NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(meshPart);
        auto sp = mi->sparsityPattern();

        SECTION("Can produce internal rowOffs and colIdx " + execName)
        {
            auto rowPtrExp = Vector<localIdx>(exec, {0, 2, 5, 8, 10});
            auto colIdxExp = Vector<localIdx>(exec, {0, 1, 0, 1, 2, 1, 2, 3, 2, 3});

            compare(sp->rowOffs(), rowPtrExp, EqualInt());
            compare(sp->colIdxs(), colIdxExp, EqualInt());
        }

        auto bsp = mi->boundarySparsityPattern();
        auto nsp = mi->nonLocalSparsityPattern();
        auto rowToDiagonalMap = la::computeRowToDiagonalMap(nsp->rowOffs(), mi);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            auto rowPtrExp = Vector<localIdx>(exec, std::vector<localIdx> {0});
            auto nlRowPtrExp = Vector<localIdx>(exec, std::vector<localIdx> {3});
            auto nlColIdxExp = Vector<localIdx>(exec, std::vector<localIdx> {4});
            auto rowToDiagMapExp = Vector<localIdx>(exec, std::vector<localIdx> {9});
            compare(bsp->rowOffs(), rowPtrExp, EqualInt());
            compare(nsp->rowOffs(), nlRowPtrExp, EqualInt());
            compare(nsp->colIdxs(), nlColIdxExp, EqualInt());
            compare(rowToDiagonalMap, rowToDiagMapExp, EqualInt());
        }
        SECTION_IF(
            mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary (no boundary) " + execName
        )
        {
            auto rowPtrExp = Vector<localIdx>(exec, {});
            auto nlRowPtrExp = Vector<localIdx>(exec, std::vector<localIdx> {0, 3});
            auto nlColIdxExp = Vector<localIdx>(exec, std::vector<localIdx> {3, 8});
            auto rowToDiagMapExp = Vector<localIdx>(exec, std::vector<localIdx> {0, 9});
            compare(bsp->rowOffs(), rowPtrExp, EqualInt());
            compare(nsp->rowOffs(), nlRowPtrExp, EqualInt());
            compare(nsp->colIdxs(), nlColIdxExp, EqualInt());
            compare(rowToDiagonalMap, rowToDiagMapExp, EqualInt());
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto rowPtrExp = Vector<localIdx>(exec, std::vector<localIdx> {3});
            auto nlRowPtrExp = Vector<localIdx>(exec, std::vector<localIdx> {0});
            auto nlColIdxExp = Vector<localIdx>(exec, std::vector<localIdx> {7});
            auto rowToDiagMapExp = Vector<localIdx>(exec, std::vector<localIdx> {0});
            compare(bsp->rowOffs(), rowPtrExp, EqualInt());
            compare(nsp->rowOffs(), nlRowPtrExp, EqualInt());
            compare(nsp->colIdxs(), nlColIdxExp, EqualInt());
            compare(rowToDiagonalMap, rowToDiagMapExp, EqualInt());
        }
    }
}

}
