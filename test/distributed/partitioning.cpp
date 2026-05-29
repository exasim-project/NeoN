// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

TEST_CASE("Distributed")
{
    // CPUExecutor only: one GPU on this machine, so a multi-rank test must not place
    // data on the GPU (ranks would contend for the single device).
    const std::string execName = "CPUExecutor";
    const Executor exec = CPUExecutor {};

    const auto nCells = 12;
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
            auto neighRanksExp = std::vector<localIdx> {1};
            auto recvIdxExp = std::vector<int> {4};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighRanksExp);
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            auto sendCountsExp = std::vector<int> {1, 0, 1, 2};
            auto neighRanksExp = std::vector<localIdx> {0, 2};
            auto recvIdxExp = std::vector<int> {3, 8};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighRanksExp);
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto sendCountsExp = std::vector<int> {0, 1, 0, 1};
            auto neighRanksExp = std::vector<localIdx> {1};
            auto recvIdxExp = std::vector<int> {7};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighRanksExp);
            REQUIRE(commPattern.sendCounts == sendCountsExp);
            REQUIRE(commPattern.recvIdx == recvIdxExp);
        }
    }

    auto mesh = create1DUniformMesh(exec, nCells);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto uVals =
        std::vector<scalar> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    auto u = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, uVals), volBCs
    );
    auto volVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    auto o = one<Vec3>();
    auto vecVals = std::vector<Vec3> {
        1 * o, 2 * o, 3 * o, 4 * o, 5 * o, 6 * o, 7 * o, 8 * o, 9 * o, 10 * o, 11 * o, 12 * o
    };
    auto vecU = finiteVolume::cellCentred::VolumeField<Vec3>(
        exec, "U", mesh, Vector<Vec3>(exec, vecVals), volVecBCs
    );

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto phiInternal =
        Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
    phi.internalVector() = phiInternal;

    auto uPart = detail::oneDPartitionField(u, meshPart, mpiEnviron);
    auto uVecPart = detail::oneDPartitionField(vecU, meshPart, mpiEnviron);
    auto phiPart = detail::oneDPartitionField(phi, meshPart, mpiEnviron);
    uPart.correctBoundaryConditions();
    uVecPart.correctBoundaryConditions();

    SECTION("Has correct partitioned VolumeField" + execName)
    {
        REQUIRE(uPart.internalVector().size() == nCells / mpiEnviron.sizeRank());
        REQUIRE(uPart.boundaryData().nBoundaries() == 2);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct " + execName)
        {
            auto uPartExp = std::vector<scalar> {1.0, 2.0, 3.0, 4.0};
            auto uVecPartExp = std::vector<Vec3> {
                1.0 * one<Vec3>(), 2.0 * one<Vec3>(), 3.0 * one<Vec3>(), 4.0 * one<Vec3>()
            };
            REQUIRE_THAT(uPart.internalVector(), Equals(uPartExp));
            REQUIRE_THAT(uVecPart.internalVector(), Equals(uVecPartExp, Approx {1e-32}));
            auto uPartBoundExp = std::vector<scalar> {0.0, 5.0};
            auto uVecPartBoundExp = std::vector<Vec3> {0.0 * one<Vec3>(), 5.0 * one<Vec3>()};
            REQUIRE_THAT(uPart.boundaryData().value(), Equals(uPartBoundExp));
            REQUIRE_THAT(uVecPart.boundaryData().value(), Equals(uVecPartBoundExp, Approx {1e-32}));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct " + execName)
        {
            auto uPartExp = std::vector<scalar> {5.0, 6.0, 7.0, 8.0};
            auto uVecPartExp = std::vector<Vec3> {
                5.0 * one<Vec3>(), 6.0 * one<Vec3>(), 7.0 * one<Vec3>(), 8.0 * one<Vec3>()
            };
            REQUIRE_THAT(uPart.internalVector(), Equals(uPartExp));
            REQUIRE_THAT(uVecPart.internalVector(), Equals(uVecPartExp, Approx {1e-32}));
            auto uPartBoundExp = std::vector<scalar> {4.0, 9.0};
            auto uVecPartBoundExp = std::vector<Vec3> {4.0 * one<Vec3>(), 9.0 * one<Vec3>()};
            REQUIRE_THAT(uPart.boundaryData().value(), Equals(uPartBoundExp));
            REQUIRE_THAT(uVecPart.boundaryData().value(), Equals(uVecPartBoundExp, Approx {1e-32}));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct " + execName)
        {
            auto uPartExp = std::vector<scalar> {9.0, 10.0, 11.0, 12.0};
            auto uVecPartExp = std::vector<Vec3> {
                9.0 * one<Vec3>(), 10.0 * one<Vec3>(), 11.0 * one<Vec3>(), 12.0 * one<Vec3>()
            };
            REQUIRE_THAT(uPart.internalVector(), Equals(uPartExp));
            REQUIRE_THAT(uVecPart.internalVector(), Equals(uVecPartExp, Approx {1e-32}));
            auto uPartBoundExp = std::vector<scalar> {0.0, 8.0};
            auto uVecPartBoundExp = std::vector<Vec3> {0.0 * one<Vec3>(), 8.0 * one<Vec3>()};
            REQUIRE_THAT(uPart.boundaryData().value(), Equals(uPartBoundExp));
            REQUIRE_THAT(uVecPart.boundaryData().value(), Equals(uVecPartBoundExp, Approx {1e-32}));
        }
    }

    SECTION("Has correct partitioned SurfaceField" + execName)
    {
        REQUIRE(phiPart.boundaryData().nBoundaries() == 2);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {1.0, 2.0, 3.0};
            REQUIRE_THAT(phiPart.internalVector(), Equals(phiExp));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {5.0, 6.0, 7.0};
            REQUIRE_THAT(phiPart.internalVector(), Equals(phiExp));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct proc boundary " + execName)
        {
            auto phiExp = std::vector<scalar> {9.0, 10.0, 11.0};
            REQUIRE_THAT(phiPart.internalVector(), Equals(phiExp));
        }
    }

    SECTION("Can produce correct sparsity pattern and face-to-matrix address" + execName)
    {
        using CsrSparsityType = NeoN::la::CsrSparsityPattern<NeoN::localIdx>;
        using CooSparsityType = NeoN::la::CooSparsityPattern<NeoN::localIdx>;

        auto [sp, mi] =
            NeoN::la::createSparsityPatternFaceToMatrixAddress<CsrSparsityType>(meshPart);

        SECTION("Can produce internal rowOffs and colIdx " + execName)
        {
            auto rowPtrExp = std::vector<localIdx> {0, 2, 5, 8, 10};
            auto colIdxExp = std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

            REQUIRE_THAT(sp->rowOffs(), Equals(rowPtrExp, EqualInt()));
            REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
        }

        const auto& diagOffset = mi->diagOffset();
        const auto& ownerOffset = mi->ownerOffset();
        const auto& neighOffset = mi->neighbourOffset();

        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct offsets " + execName)
        {
            REQUIRE_THAT(diagOffset, Equals(I({0, 1, 1, 1}), EqualInt()));
            REQUIRE_THAT(ownerOffset, Equals(I({1, 2, 2}), EqualInt()));
            REQUIRE_THAT(neighOffset, Equals(I({0, 0, 0}), EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct offsets " + execName)
        {
            REQUIRE_THAT(diagOffset, Equals(I({0, 1, 1, 1}), EqualInt()));
            REQUIRE_THAT(ownerOffset, Equals(I({1, 2, 2}), EqualInt()));
            REQUIRE_THAT(neighOffset, Equals(I({0, 0, 0}), EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct offsets " + execName)
        {
            REQUIRE_THAT(diagOffset, Equals(I({0, 1, 1, 1}), EqualInt()));
            REQUIRE_THAT(ownerOffset, Equals(I({1, 2, 2}), EqualInt()));
            REQUIRE_THAT(neighOffset, Equals(I({0, 0, 0}), EqualInt()));
        }

        auto bsp = NeoN::la::createBoundarySparsityPattern<CooSparsityType>(meshPart, *mi);
        SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 has correct boundary sparsity " + execName)
        {
            REQUIRE_THAT(bsp->rowIdxs(), Equals(I({0}), EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 has correct boundary sparsity " + execName)
        {
            REQUIRE_THAT(bsp->rowIdxs(), Equals(std::vector<localIdx> {}, EqualInt()));
        }
        SECTION_IF(mpiEnviron.rank() == 2, "Rank 2 has correct boundary sparsity " + execName)
        {
            REQUIRE_THAT(bsp->rowIdxs(), Equals(I({3}), EqualInt()));
        }
    }
}

}
