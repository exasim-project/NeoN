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
    auto [execName, exec] = GENERATE(allAvailableExecutor());

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

        // [T1] Regression test for commPattern audit N-H2: isDistributed() must be the
        // authoritative dispatch question, returning true on every rank of a multi-rank
        // job — independent of whether the local sendCounts/recvIdx are non-empty.
        // Previously the dispatch used `!sendCounts.empty()`, which silently returned
        // false on any rank that happened to have no processor faces, deadlocking the
        // job when peers entered distributed collectives that this rank never joined.
        SECTION("isDistributed() agrees across ranks " + execName)
        {
            const int local = commPattern.isDistributed() ? 1 : 0;
            int globalAnd = 0;
            int globalOr = 0;
            MPI_Allreduce(&local, &globalAnd, 1, MPI_INT, MPI_BAND, mpiEnviron.comm());
            MPI_Allreduce(&local, &globalOr, 1, MPI_INT, MPI_BOR, mpiEnviron.comm());
            // Every rank must answer the same question identically.
            REQUIRE(globalAnd == globalOr);
            // On a 3-rank job, every rank must dispatch to the distributed path.
            REQUIRE(local == 1);
        }

        // [T1.b] Pattern with no local proc faces but `isPartitioned` set must
        // still dispatch to the distributed path. This simulates the rare Scotch
        // corner case where a rank's partition is bounded entirely by physical
        // patches; that rank still has to enter every distributed collective its
        // peers do, otherwise the job deadlocks. `isPartitioned` is the
        // dispatch-determining bit, set globally via MPI_Allreduce in
        // `computeCommunicationPattern` so that all ranks agree.
        SECTION(
            "isDistributed() true for partitioned pattern with empty local sendCounts " + execName
        )
        {
            CommunicationPattern empty;
            empty.env = mpiEnviron;
            empty.isPartitioned = true; // as if computeCommunicationPattern set it
            REQUIRE(empty.sendCounts.empty());
            REQUIRE(empty.recvIdx.empty());
            REQUIRE(empty.isDistributed() == true);
        }

        // [T1.c] A default-constructed pattern (no Allreduce, no mesh hint) is
        // treated as non-distributed. This corresponds to a multi-rank job
        // carrying a LinearSystem built from a per-rank full copy of the global
        // mesh (e.g. the canonical local `ls` used as a sanity-check baseline
        // against `lsDst` in the operator test): every rank must take the local
        // solve branch.
        SECTION("isDistributed() false for default-constructed pattern " + execName)
        {
            CommunicationPattern defaulted;
            REQUIRE(defaulted.isDistributed() == false);
        }

        // [T3] Symmetric round-trip via the pattern: every value in recvIdx on this
        // rank must equal a value the neighbour sent. We re-send the global cell
        // indices that we shipped originally and confirm receivers get back what we
        // claimed to send. Together with the rank-by-rank recvIdx assertions above,
        // this catches one-side malformed patterns.
        SECTION("Pattern round-trips global cell ids " + execName)
        {
            const int nRanks = mpiEnviron.sizeRank();
            const auto globalOffset = static_cast<int>(meshPart.globalOffset());

            // sendCounts[0..nRanks) are face counts per destination rank.
            std::vector<int> sendCounts(
                commPattern.sendCounts.begin(), commPattern.sendCounts.begin() + nRanks
            );
            std::vector<int> sdispl(nRanks, 0);
            for (int r = 1; r < nRanks; ++r)
            {
                sdispl[r] = sdispl[r - 1] + sendCounts[r - 1];
            }
            const int totalSend = sdispl.back() + sendCounts.back();

            // Build the send buffer in the SAME ordering the pattern uses
            // internally: walk proc faces in patch order, dispatched by neighbour
            // rank. Each rank announces "I am sending you my global cell id X".
            std::vector<int> sendBuf(totalSend, 0);
            const auto& neighbourRanks = meshPart.boundaryMesh().neighbourRank();
            const auto& offsets = meshPart.boundaryMesh().offset();
            const auto nInner = meshPart.boundaryMesh().nBoundaries()
                              - meshPart.boundaryMesh().nProcBoundaryPatches();
            const auto faceCellsH = meshPart.boundaryMesh().faceOwners().copyToHost();
            const auto procStart = offsets[static_cast<std::size_t>(nInner)];
            std::vector<int> cursor(nRanks, 0);
            for (std::size_t i = 0; i < neighbourRanks.size(); ++i)
            {
                const int dst = static_cast<int>(neighbourRanks[i]);
                const auto patchStart = offsets[static_cast<std::size_t>(nInner + i)];
                const auto patchEnd = offsets[static_cast<std::size_t>(nInner + i + 1)];
                for (auto k = patchStart; k < patchEnd; ++k)
                {
                    const int slot = sdispl[dst] + cursor[dst]++;
                    sendBuf[slot] = static_cast<int>(faceCellsH.view()[k]) + globalOffset;
                }
            }

            std::vector<int> recvCounts(nRanks, 0);
            mpi::allToAll<int>(sendCounts.data(), 1, recvCounts.data(), 1, mpiEnviron.comm());
            std::vector<int> rdispl(nRanks, 0);
            for (int r = 1; r < nRanks; ++r)
            {
                rdispl[r] = rdispl[r - 1] + recvCounts[r - 1];
            }
            const int totalRecv = rdispl.back() + recvCounts.back();
            std::vector<int> recvBuf(totalRecv, 0);
            mpi::allToAllV<int>(
                sendBuf.data(),
                sendCounts.data(),
                sdispl.data(),
                recvBuf.data(),
                recvCounts.data(),
                rdispl.data(),
                mpiEnviron.comm()
            );

            // The reconstructed recv buffer must equal the pattern's stored recvIdx,
            // proving that what the neighbour announces matches what we recorded.
            REQUIRE(recvBuf == commPattern.recvIdx);
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
