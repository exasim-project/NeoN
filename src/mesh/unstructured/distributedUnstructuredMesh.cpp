// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/** @brief Creates a 1D uniform mesh partition for the current MPI rank.
 *
 *  The global domain spans [0, 1].  Each rank owns a contiguous slab of
 *  nCells cells.  Regular (physical) boundary patches appear on the domain
 *  boundary; processor patches are created where partitions share a face.
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells)
{
    mpi::Environment mpiEnviron;
    scalar rightBoundary {static_cast<scalar>(1.0) / mpiEnviron.sizeRank()};

    const bool isFirst = mpiEnviron.rank() == 0;
    const bool isLast = mpiEnviron.rank() == mpiEnviron.sizeRank() - 1;
    const localIdx nProcBoundaryFaces = (isFirst || isLast) ? 1 : 2;
    const localIdx nRegularBoundary = static_cast<localIdx>((isFirst || isLast) ? 1 : 0);

    // boundary face owners and weights (regular patches first, proc patches after)
    auto faceCellVec = std::vector<localIdx>();
    auto boundaryWeights = std::vector<scalar>();
    auto neighRanks = std::vector<localIdx>();

    if (!isFirst && !isLast)
    {
        // middle rank: only proc patches (no regular boundary)
        faceCellVec = {0, nCells - 1};
        boundaryWeights = {-1.0, 1.0};
        neighRanks = {
            static_cast<localIdx>(mpiEnviron.rank() - 1),
            static_cast<localIdx>(mpiEnviron.rank() + 1)
        };
    }
    else if (isFirst)
    {
        // rank 0: xmin regular then proc right
        faceCellVec = {0, nCells - 1};
        boundaryWeights = {-1.0, 1.0};
        neighRanks = {static_cast<localIdx>(1)};
    }
    else
    {
        // last rank: xmax regular first then proc left
        faceCellVec = {nCells - 1, 0};
        boundaryWeights = {1.0, -1.0};
        neighRanks = {static_cast<localIdx>(mpiEnviron.rank() - 1)};
    }

    labelVector faceCells(exec, faceCellVec);

    // Face geometry data for each boundary slot (2 slots regardless of rank).
    // For the last rank the boundary face ordering is [xmax, proc-left] — the
    // face data vectors are in that order too (swapped relative to rank-0/mid).
    std::vector<Vec3> bcFaceCentersVec {{0.0, 0.0, 0.0}, {rightBoundary, 0.0, 0.0}};
    std::vector<Vec3> bcFaceNormalsVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<scalar> bcFaceAreasVec {1.0, 1.0};
    std::vector<Vec3> bcFaceUnitNormalsVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    if (isLast)
    {
        std::swap(bcFaceCentersVec[0], bcFaceCentersVec[1]);
        std::swap(bcFaceNormalsVec[0], bcFaceNormalsVec[1]);
        std::swap(bcFaceAreasVec[0], bcFaceAreasVec[1]);
        std::swap(bcFaceUnitNormalsVec[0], bcFaceUnitNormalsVec[1]);
    }

    auto baseMesh = create1DUniformMesh(exec, nCells, rightBoundary);

    BoundaryMesh boundaryMesh(
        exec,
        faceCells,
        {exec, bcFaceCentersVec},
        baseMesh.boundaryMesh().ownerCellCenters(),
        {exec, bcFaceNormalsVec},
        {exec, bcFaceAreasVec},
        {exec, bcFaceUnitNormalsVec},
        baseMesh.boundaryMesh().delta(),
        {exec, boundaryWeights},
        baseMesh.boundaryMesh().deltaCoeffs(),
        {0, 1, 2}, // offset: 2 patches
        nProcBoundaryFaces,
        neighRanks
    );

    // For the last rank, the baseMesh boundary faces need to be physically swapped so
    // that regular (xmax) comes before the proc face in the internal face arrays.
    UnstructuredMesh mesh(
        exec,
        baseMesh.points(),
        baseMesh.cellVolumes(),
        baseMesh.cellCenters(),
        baseMesh.faceNormals(),
        baseMesh.faceCenters(),
        baseMesh.faceAreas(),
        baseMesh.faceOwners(),
        baseMesh.faceNeighbors(),
        std::move(boundaryMesh)
    );

    return mesh;
}

/** @brief See declaration in communicationPattern.hpp for full documentation.
 *
 *  Implementation outline:
 *  1. Build `sendCounts[r]` = faces this rank sends to rank r.
 *  2. Build `buffer` of global cell indices (local owner cell + globalOffset)
 *     for every processor boundary face.
 *  3. Exchange `sendCounts` via `allToAll` to learn `recvCounts` per rank.
 *  4. Exchange the global cell index buffer via `allToAllV` to populate
 *     `recvIdx` — the list of global cells this rank will receive.
 *  5. Return a `CommunicationPattern` with the collected data.
 */
CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh)
{
    mpi::Environment mpiEnviron;
    if (!mesh.boundaryMesh().isDistributed())
    {
        return {};
    }

    auto sendCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank() + 1), 0);

    const auto& neighbourRanks = mesh.boundaryMesh().neighbourRank();
    const auto& offsets = mesh.boundaryMesh().offset();
    const auto nInnerBoundaries =
        mesh.boundaryMesh().nBoundaries() - mesh.boundaryMesh().nProcBoundaryPatches();

    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        const auto patchSize = static_cast<int>(
            offsets[static_cast<std::size_t>(nInnerBoundaries + i + 1)]
            - offsets[static_cast<std::size_t>(nInnerBoundaries + i)]
        );
        sendCounts[static_cast<std::size_t>(targetRank)] = patchSize;
        sendCounts[static_cast<std::size_t>(mpiEnviron.sizeRank())] += patchSize;
    }

    const auto globalOffset = mesh.globalOffset();
    const auto faceCellsH = mesh.boundaryMesh().faceOwners().copyToHost();

    auto buffer = std::vector<localIdx>();
    buffer.reserve(static_cast<std::size_t>(mesh.boundaryMesh().nProcBoundaryFaces()));
    const auto procStart = offsets[static_cast<std::size_t>(nInnerBoundaries)];
    for (int i = 0; i < mesh.boundaryMesh().nProcBoundaryFaces(); i++)
    {
        buffer.push_back(faceCellsH.view()[i + procStart] + globalOffset);
    }

    auto sdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        sdispl[static_cast<std::size_t>(targetRank)] =
            static_cast<int>(offsets[static_cast<std::size_t>(nInnerBoundaries + i)] - procStart);
    }

    auto recvCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    mpi::allToAll<int>(sendCounts.data(), 1, recvCounts.data(), 1, mpiEnviron.comm());

    auto rdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int r = 1; r < mpiEnviron.sizeRank(); ++r)
    {
        rdispl[static_cast<std::size_t>(r)] =
            rdispl[static_cast<std::size_t>(r - 1)] + recvCounts[static_cast<std::size_t>(r - 1)];
    }
    int totalRecv = rdispl.back() + recvCounts.back();

    auto recvIdx = std::vector<int>(static_cast<std::size_t>(totalRecv));

    std::vector<int> sendBuffer(buffer.size());
    for (std::size_t i = 0; i < buffer.size(); ++i)
        sendBuffer[i] = static_cast<int>(buffer[i]);

    mpi::allToAllV<int>(
        sendBuffer.data(),
        sendCounts.data(),
        sdispl.data(),
        recvIdx.data(),
        recvCounts.data(),
        rdispl.data(),
        mpiEnviron.comm()
    );

    std::vector<localIdx> boundaryMapVector;
    return CommunicationPattern {sendCounts, recvIdx, boundaryMapVector, mpiEnviron};
}

} // namespace NeoN
