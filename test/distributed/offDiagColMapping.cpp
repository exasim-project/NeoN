// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace NeoN
{

// On a 2x2 partition every rank owns TWO processor patches (to two different
// neighbour ranks), ordered geometrically [xmin, xmax, ymin, ymax]. The off-diagonal
// sparsity colIdx[pfacei] must reference a global cell that lives on the neighbour
// rank that local processor face pfacei actually connects to. computeCommunicationPattern
// returns recvIdx grouped by ascending SOURCE rank, while local proc faces are in
// proc-patch (geometric) order; if the two orderings disagree the off-diagonal columns
// are swapped between patches. This test pins that.
TEST_CASE("OffDiagonal column mapping (2x2 multi-patch)")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx totalRanks = 4;
    const localIdx nCells = 4; // per part, so global cells per rank = nCells*nCells

    NeoN::mpi::Environment mpiEnviron;
    const auto rank = static_cast<localIdx>(mpiEnviron.rank());
    if (mpiEnviron.sizeRank() != totalRanks) return;

    auto meshPart = create2DUniformMeshPart(exec, nCells, totalRanks, rank);

    const localIdx nCellsPerRank = nCells * nCells;

    // Build the off-diagonal sparsity exactly as the assembler does.
    auto [sparsity, faceToMatrixAddr] =
        NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::la::CsrSparsityPattern<localIdx>>(
            meshPart
        );
    auto offDiag =
        NeoN::la::createOffDiagonalSparsityPattern<NeoN::la::CsrSparsityPattern<localIdx>>(
            meshPart, *faceToMatrixAddr
        );

    auto colIdxH = offDiag->colIdxs().copyToHost();
    const auto colIdx = colIdxH.view();

    // colIdxs are stored in row-sorted order (ascending faceOwner). Use rowOrderWriteIndex
    // to translate from original proc-face index to its sorted position.
    auto rowOrderVec = meshPart.boundaryMesh().getRowOrderWriteIndex().copyToHost();
    const auto rowOrder = rowOrderVec.view();

    const auto nProcFaces = meshPart.nProcBoundaryFaces();
    const auto nBoundaryFaces = meshPart.nBoundaryFaces();
    const auto& offsets = meshPart.boundaryMesh().offset();
    const auto& neighRanks = meshPart.boundaryMesh().neighbourRank();
    const auto nProcPatches = meshPart.boundaryMesh().nProcBoundaryPatches();
    const auto nInnerBoundaries = meshPart.boundaryMesh().nBoundaries() - nProcPatches;

    SECTION("colIdx of each proc face lands on the patch's true neighbour rank " + execName)
    {
        // For each local processor face, determine the neighbour rank from its patch
        // (proc-patch / offset order), then assert colIdx falls in that rank's global
        // cell range [neigh*nCellsPerRank, (neigh+1)*nCellsPerRank).
        // colIdxs are stored in row-sorted order, so use rowOrder[pfacei] to look up
        // the column index for original proc face pfacei.
        for (localIdx p = 0; p < nProcPatches; ++p)
        {
            const auto patchStart = offsets[static_cast<std::size_t>(nInnerBoundaries + p)];
            const auto patchEnd = offsets[static_cast<std::size_t>(nInnerBoundaries + p + 1)];
            const auto neigh = neighRanks[static_cast<std::size_t>(p)];
            const auto lo = neigh * nCellsPerRank;
            const auto hi = (neigh + 1) * nCellsPerRank;
            // proc faces are laid out [nBoundaryFaces, nBoundaryFaces + nProcFaces);
            // the patch's proc-face indices are [patchStart - nBoundaryFaces, patchEnd -
            // nBoundaryFaces)
            for (localIdx bf = patchStart; bf < patchEnd; ++bf)
            {
                const localIdx pfacei = bf - nBoundaryFaces;
                const auto col = colIdx[rowOrder[pfacei]];
                INFO(
                    "rank " << rank << " patch " << p << " neigh " << neigh << " pfacei " << pfacei
                            << " colIdx " << col << " expected in [" << lo << "," << hi << ")"
                );
                REQUIRE(col >= lo);
                REQUIRE(col < hi);
            }
        }
    }
    (void)nProcFaces;
}

}
