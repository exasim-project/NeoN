// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/distributed/haloExchange.hpp"

#ifdef NF_WITH_MPI_SUPPORT

namespace NeoN
{

// 3-rank MPI test for CommunicationPattern::boundaryMapVector.
//
// Asserts two criteria for the populated scatter map:
//   (1) boundaryMapVector is non-empty on a rank that has processor boundary faces.
//   (2) boundaryMapVector is a bijection over [0, nProcBoundaryFaces) — i.e. it
//       is a valid permutation, so a scatter driven by it writes every proc face
//       index exactly once.  A defective map (e.g. all entries == 0) passes the
//       non-empty check but fails the bijection check.
//
// Mesh: 1D uniform partition, 4 cells per rank, 3 ranks (MPI_SIZE 3).
// On every rank the partition shares at least one processor boundary face with a
// neighbour, so nProcBoundaryFaces() > 0 for all ranks.
TEST_CASE("CommunicationPattern boundaryMapVector populated and scatter-correct", "[HALO-01]")
{
    mpi::Environment mpiEnviron;
    const Executor exec = CPUExecutor {};

    const localIdx nLocal = 4;
    auto mesh = create1DUniformMeshPart(exec, nLocal);

    auto pattern = computeCommunicationPattern(mesh);

    const auto nPF = mesh.nProcBoundaryFaces();

    // Precondition: every rank in a 1D 3-rank partition has at least one proc face.
    REQUIRE(nPF > 0);

    // Criterion 1: non-empty.
    REQUIRE(!pattern.boundaryMapVector.empty());

    // Criterion 1b: size must equal the total number of received proc-face entries
    // (one per proc boundary face — same as nProcBoundaryFaces for a 1D partition).
    REQUIRE(static_cast<localIdx>(pattern.boundaryMapVector.size()) == static_cast<localIdx>(nPF));

    // Criterion 1c: all values lie in [0, nProcBoundaryFaces) (scatter-target invariant).
    for (const auto v : pattern.boundaryMapVector)
    {
        REQUIRE(v >= 0);
        REQUIRE(v < static_cast<localIdx>(nPF));
    }

    // Criterion 2: scatter-correct — boundaryMapVector must be a permutation of
    // [0, nProcBoundaryFaces).  A scatter that wrote everything to index 0 would
    // pass criterion 1 but fail here.
    {
        std::vector<int> seen(static_cast<std::size_t>(nPF), 0);
        for (const auto v : pattern.boundaryMapVector)
            seen[static_cast<std::size_t>(v)]++;
        for (const auto c : seen)
            REQUIRE(c == 1); // bijection: rank-grouped position <-> proc-face index
    }

    // Value-scatter through the unified primitive. Each rank sends its own rank id on every proc
    // face, so the neighbour receives — at each of its proc faces — the rank id of the cell on the
    // far side, which must equal that face's neighbour rank. This proves the rank-grouped recv
    // buffer is scattered back onto the correct proc face: a collapsed/identity scatter would leave
    // a face holding the -1 sentinel or the wrong neighbour's id.
    {
        const auto myRank = static_cast<scalar>(mpiEnviron.rank());
        Vector<scalar> sendV(exec, nPF, myRank);
        Vector<scalar> recvV(exec, nPF, scalar {-1});
        haloExchange<scalar>(exec, mesh, sendV.data(), recvV.data(), pattern);

        // Expected neighbour rank per proc face, in proc-patch (offset) order — the same order
        // haloExchange assigns to proc-face indices via boundaryMapVector.
        const auto& neighbourRanks = mesh.boundaryMesh().neighbourRank();
        const auto& offsets = mesh.boundaryMesh().offset();
        const auto nInner =
            mesh.boundaryMesh().nBoundaries() - mesh.boundaryMesh().nProcBoundaryPatches();
        std::vector<scalar> expected(static_cast<std::size_t>(nPF));
        localIdx f = 0;
        for (std::size_t p = 0; p < neighbourRanks.size(); ++p)
        {
            const auto sz = offsets[static_cast<std::size_t>(nInner) + p + 1]
                          - offsets[static_cast<std::size_t>(nInner) + p];
            for (localIdx k = 0; k < sz; ++k)
                expected[static_cast<std::size_t>(f++)] = static_cast<scalar>(neighbourRanks[p]);
        }

        auto recvHost = recvV.copyToHost();
        const auto rv = recvHost.view();
        for (localIdx i = 0; i < nPF; ++i)
            REQUIRE(rv[i] == Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1e-12));
    }
}

} // namespace NeoN

#endif // NF_WITH_MPI_SUPPORT
