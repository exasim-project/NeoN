// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace NeoN
{

TEST_CASE("Distributed2D")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 2 x 2 partitioning of the unit square: totalRanks == ranksXPart^2, ranksXPart == ranksYPart.
    const localIdx totalRanks = 4;
    const localIdx nCells = 4; // nCells x nCells cells per part

    NeoN::mpi::Environment mpiEnviron;
    const auto rank = static_cast<localIdx>(mpiEnviron.rank());

    auto meshPart = create2DUniformMeshPart(exec, nCells, totalRanks, rank);

    SECTION("Has correct partitioned 2d mesh " + execName)
    {
        // nCells x nCells interior cells
        REQUIRE(meshPart.nCells() == nCells * nCells);
        // internal faces: x-normal (nCells-1)*nCells + y-normal nCells*(nCells-1)
        REQUIRE(meshPart.nInternalFaces() == 2 * nCells * (nCells - 1));
        REQUIRE(meshPart.boundaryMesh().isDistributed());

        // Every rank of a 2 x 2 grid is a corner: two regular and two processor patches.
        REQUIRE(meshPart.boundaryMesh().nBoundaries() == 4);
        REQUIRE(meshPart.boundaryMesh().nProcBoundaryPatches() == 2);
        REQUIRE(meshPart.nProcBoundaryFaces() == 2 * nCells);
        REQUIRE(meshPart.nBoundaryFaces() == 2 * nCells);
    }

    SECTION("Has correct processor neighbours " + execName)
    {
        // Processor patches are ordered following the base patch order [xmin, xmax, ymin, ymax],
        // restricted to the shared (processor) sides.  rankX = rank % 2, rankY = rank / 2.
        SECTION_IF(rank == 0, "Rank 0 neighbours " + execName)
        {
            // xmax -> 1, ymax -> 2
            auto neighExp = std::vector<localIdx> {1, 2};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighExp);
        }
        SECTION_IF(rank == 1, "Rank 1 neighbours " + execName)
        {
            // xmin -> 0, ymax -> 3
            auto neighExp = std::vector<localIdx> {0, 3};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighExp);
        }
        SECTION_IF(rank == 2, "Rank 2 neighbours " + execName)
        {
            // xmax -> 3, ymin -> 0
            auto neighExp = std::vector<localIdx> {3, 0};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighExp);
        }
        SECTION_IF(rank == 3, "Rank 3 neighbours " + execName)
        {
            // xmin -> 2, ymin -> 1
            auto neighExp = std::vector<localIdx> {2, 1};
            REQUIRE(meshPart.boundaryMesh().neighbourRank() == neighExp);
        }
    }

    SECTION("Can create a communication pattern " + execName)
    {
        auto commPattern = computeCommunicationPattern(meshPart);
        // Each rank sends nCells faces to each of its two neighbours.
        REQUIRE(commPattern.sendCounts.back() == 2 * nCells);
    }
}

}
