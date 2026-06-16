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

    // Place this rank's slab at its true position in the global [0, 1] domain. A faithful
    // decomposition shares one global coordinate system across ranks (as decomposePar produces),
    // which is required for any cross-rank coordinate exchange — e.g. the geometry scheme's
    // exact processor-face owner-to-neighbour distance |Cnei - Cown|. Without the offset every
    // rank would sit at [0, rightBoundary] and neighbour cell centres would be in the wrong frame.
    const scalar xOffset = static_cast<scalar>(mpiEnviron.rank()) * rightBoundary;

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
    std::vector<Vec3> bcFaceCentersVec {{xOffset, 0.0, 0.0}, {xOffset + rightBoundary, 0.0, 0.0}};
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

    // Shift the rank-local mesh geometry into its global position (see xOffset above). Only
    // positions move; areas, normals, volumes and cell-to-cell / cell-to-face differences are
    // unchanged, so frame-independent quantities (fluxes, Courant number, gradients, projected
    // distances) are unaffected.
    auto shiftXInPlace = [&](Vector<Vec3>& v)
    {
        auto h = v.copyToHost();
        auto hv = h.view();
        for (localIdx i = 0; i < h.size(); ++i)
            hv[i][0] += xOffset;
        v = h.copyToExecutor(exec);
    };
    shiftXInPlace(baseMesh.points());
    shiftXInPlace(baseMesh.cellCenters());
    shiftXInPlace(baseMesh.faceCenters());

    // Owner-cell centres held on the boundary mesh need the same shift (const accessor → copy).
    auto ownerCellCentersH = baseMesh.boundaryMesh().ownerCellCenters().copyToHost();
    {
        auto v = ownerCellCentersH.view();
        for (localIdx i = 0; i < ownerCellCentersH.size(); ++i)
            v[i][0] += xOffset;
    }
    auto ownerCellCentersGlobal = ownerCellCentersH.copyToExecutor(exec);

    BoundaryMesh boundaryMesh(
        exec,
        faceCells,
        {exec, bcFaceCentersVec},
        ownerCellCentersGlobal,
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

/** @brief Builds the local 2D mesh partition for a given rank.
 *
 *  The global domain is the unit square [0,1] x [0,1] partitioned into a
 *  ranksXPart x ranksYPart grid of square subdomains (ranksXPart == ranksYPart,
 *  totalRanks == ranksXPart * ranksXPart).  Each part is an nCells x nCells
 *  hex slab (one cell thick in z).  Boundary patches that sit on the global
 *  domain boundary remain regular; patches shared with a neighbouring partition
 *  become processor patches.  As required by BoundaryMesh, regular patches are
 *  stored before processor patches.
 */
UnstructuredMesh create2DUniformMeshPart(
    const Executor exec, const localIdx nCells, const localIdx totalRanks, const localIdx rank
)
{
    // totalRanks == ranksXPart * ranksXPart and ranksXPart == ranksYPart.
    localIdx ranksXPart = 1;
    while (ranksXPart * ranksXPart < totalRanks)
    {
        ++ranksXPart;
    }
    NF_ASSERT(
        ranksXPart * ranksXPart == totalRanks,
        "create2DUniformMeshPart requires totalRanks to be a perfect square"
    );
    const localIdx ranksYPart = ranksXPart;

    const localIdx rankX = rank % ranksXPart;
    const localIdx rankY = rank / ranksXPart;

    // Each part is a square subdomain so that the parts tile the unit square.
    const scalar lx = static_cast<scalar>(1.0) / static_cast<scalar>(ranksXPart);
    const scalar ly = static_cast<scalar>(1.0) / static_cast<scalar>(ranksYPart);

    auto baseMesh = create2DUniformMesh(exec, nCells, nCells, lx, ly);

    // The base 2D mesh has four boundary patches, each with nCells faces, in the
    // order [xmin, xmax, ymin, ymax].
    const localIdx nFacesPerPatch = nCells;
    const localIdx nInternalFaces = baseMesh.nInternalFaces();

    // Classify each patch as regular (on the global domain boundary) or processor
    // (shared with a neighbouring partition).  normalSign encodes the outward
    // normal direction along the patch axis, matching create1DUniformMeshPart.
    struct PatchInfo
    {
        localIdx baseIndex; // patch index in base mesh order [xmin, xmax, ymin, ymax]
        bool isProc;
        localIdx neighRank;
        scalar normalSign;
    };

    const PatchInfo patches[4] = {
        {0, rankX != 0, rank - 1, -1.0},                     // xmin
        {1, rankX != ranksXPart - 1, rank + 1, 1.0},         // xmax
        {2, rankY != 0, rank - ranksXPart, -1.0},            // ymin
        {3, rankY != ranksYPart - 1, rank + ranksXPart, 1.0} // ymax
    };

    // Order patches so that regular patches come first and processor patches last.
    std::vector<PatchInfo> ordered;
    ordered.reserve(4);
    auto neighRanks = std::vector<localIdx>();
    for (const auto& patch : patches)
    {
        if (!patch.isProc) ordered.push_back(patch);
    }
    for (const auto& patch : patches)
    {
        if (patch.isProc)
        {
            ordered.push_back(patch);
            neighRanks.push_back(patch.neighRank);
        }
    }
    const localIdx nProcPatches = static_cast<localIdx>(neighRanks.size());

    // Gather the per-face boundary data from the base mesh into the new patch order.
    auto bOwnersH = baseMesh.boundaryMesh().faceOwners().copyToHost();
    auto bCentersH = baseMesh.boundaryMesh().faceCenters().copyToHost();
    auto bOwnerCCH = baseMesh.boundaryMesh().ownerCellCenters().copyToHost();
    auto bNormalsH = baseMesh.boundaryMesh().faceNormals().copyToHost();
    auto bAreasH = baseMesh.boundaryMesh().faceAreas().copyToHost();
    auto bUnitNormalsH = baseMesh.boundaryMesh().faceUnitNormals().copyToHost();
    auto bDeltaH = baseMesh.boundaryMesh().delta().copyToHost();
    auto bDeltaCoeffsH = baseMesh.boundaryMesh().deltaCoeffs().copyToHost();

    const auto bOwners = bOwnersH.view();
    const auto bCenters = bCentersH.view();
    const auto bOwnerCC = bOwnerCCH.view();
    const auto bNormals = bNormalsH.view();
    const auto bAreas = bAreasH.view();
    const auto bUnitNormals = bUnitNormalsH.view();
    const auto bDelta = bDeltaH.view();
    const auto bDeltaCoeffs = bDeltaCoeffsH.view();

    const auto nBoundaryFaces = static_cast<std::size_t>(4 * nFacesPerPatch);
    auto faceOwnersVec = std::vector<label>();
    auto faceCentersVec = std::vector<Vec3>();
    auto ownerCentersVec = std::vector<Vec3>();
    auto faceNormalsVec = std::vector<Vec3>();
    auto faceAreasVec = std::vector<scalar>();
    auto faceUnitNormalsVec = std::vector<Vec3>();
    auto deltaVec = std::vector<Vec3>();
    auto weightsVec = std::vector<scalar>();
    auto deltaCoeffsVec = std::vector<scalar>();
    faceOwnersVec.reserve(nBoundaryFaces);
    faceCentersVec.reserve(nBoundaryFaces);
    ownerCentersVec.reserve(nBoundaryFaces);
    faceNormalsVec.reserve(nBoundaryFaces);
    faceAreasVec.reserve(nBoundaryFaces);
    faceUnitNormalsVec.reserve(nBoundaryFaces);
    deltaVec.reserve(nBoundaryFaces);
    weightsVec.reserve(nBoundaryFaces);
    deltaCoeffsVec.reserve(nBoundaryFaces);

    for (const auto& patch : ordered)
    {
        const localIdx start = patch.baseIndex * nFacesPerPatch;
        for (localIdx f = 0; f < nFacesPerPatch; ++f)
        {
            const localIdx bi = start + f;
            faceOwnersVec.push_back(bOwners[bi]);
            faceCentersVec.push_back(bCenters[bi]);
            ownerCentersVec.push_back(bOwnerCC[bi]);
            faceNormalsVec.push_back(bNormals[bi]);
            faceAreasVec.push_back(bAreas[bi]);
            faceUnitNormalsVec.push_back(bUnitNormals[bi]);
            deltaVec.push_back(bDelta[bi]);
            weightsVec.push_back(patch.normalSign);
            deltaCoeffsVec.push_back(bDeltaCoeffs[bi]);
        }
    }

    // Four patches, each nFacesPerPatch faces.
    auto offset = std::vector<localIdx> {
        0, nFacesPerPatch, 2 * nFacesPerPatch, 3 * nFacesPerPatch, 4 * nFacesPerPatch
    };

    BoundaryMesh boundaryMesh(
        exec,
        {exec, faceOwnersVec},
        {exec, faceCentersVec},
        {exec, ownerCentersVec},
        {exec, faceNormalsVec},
        {exec, faceAreasVec},
        {exec, faceUnitNormalsVec},
        {exec, deltaVec},
        {exec, weightsVec},
        {exec, deltaCoeffsVec},
        offset,
        nProcPatches,
        neighRanks
    );

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

/** @brief Builds the local 2D mesh partition for a given rank using x-only decomposition.
 *
 *  The global domain is the unit square [0,1] x [0,1] partitioned into
 *  totalRanks x 1 strips along x.  Each part is an nCells x nCells hex slab
 *  (one cell thick in z) of width 1/totalRanks.  ymin and ymax patches are
 *  always regular; xmin is regular only on rank 0 and xmax only on the last rank.
 */
UnstructuredMesh create2DUniformMeshPartX(
    const Executor exec, const localIdx nCells, const localIdx totalRanks, const localIdx rank
)
{
    const localIdx ranksXPart = totalRanks;

    const scalar lx = static_cast<scalar>(1.0) / static_cast<scalar>(ranksXPart);
    const scalar ly = static_cast<scalar>(1.0);

    auto baseMesh = create2DUniformMesh(exec, nCells, nCells, lx, ly);

    const localIdx nFacesPerPatch = nCells;
    const localIdx nInternalFaces = baseMesh.nInternalFaces();

    struct PatchInfo
    {
        localIdx baseIndex;
        bool isProc;
        localIdx neighRank;
        scalar normalSign;
    };

    const bool isFirst = (rank == 0);
    const bool isLast = (rank == ranksXPart - 1);

    const PatchInfo patches[4] = {
        {0, !isFirst, rank - 1, -1.0}, // xmin
        {1, !isLast, rank + 1, 1.0},   // xmax
        {2, false, -1, -1.0},          // ymin (always regular)
        {3, false, -1, 1.0},           // ymax (always regular)
    };

    std::vector<PatchInfo> ordered;
    ordered.reserve(4);
    auto neighRanks = std::vector<localIdx>();
    for (const auto& patch : patches)
        if (!patch.isProc) ordered.push_back(patch);
    for (const auto& patch : patches)
    {
        if (patch.isProc)
        {
            ordered.push_back(patch);
            neighRanks.push_back(patch.neighRank);
        }
    }
    const localIdx nProcPatches = static_cast<localIdx>(neighRanks.size());

    auto bOwnersH = baseMesh.boundaryMesh().faceOwners().copyToHost();
    auto bCentersH = baseMesh.boundaryMesh().faceCenters().copyToHost();
    auto bOwnerCCH = baseMesh.boundaryMesh().ownerCellCenters().copyToHost();
    auto bNormalsH = baseMesh.boundaryMesh().faceNormals().copyToHost();
    auto bAreasH = baseMesh.boundaryMesh().faceAreas().copyToHost();
    auto bUnitNormalsH = baseMesh.boundaryMesh().faceUnitNormals().copyToHost();
    auto bDeltaH = baseMesh.boundaryMesh().delta().copyToHost();
    auto bDeltaCoeffsH = baseMesh.boundaryMesh().deltaCoeffs().copyToHost();

    const auto bOwners = bOwnersH.view();
    const auto bCenters = bCentersH.view();
    const auto bOwnerCC = bOwnerCCH.view();
    const auto bNormals = bNormalsH.view();
    const auto bAreas = bAreasH.view();
    const auto bUnitNormals = bUnitNormalsH.view();
    const auto bDelta = bDeltaH.view();
    const auto bDeltaCoeffs = bDeltaCoeffsH.view();

    const auto nBoundaryFaces = static_cast<std::size_t>(4 * nFacesPerPatch);
    auto faceOwnersVec = std::vector<label>();
    auto faceCentersVec = std::vector<Vec3>();
    auto ownerCentersVec = std::vector<Vec3>();
    auto faceNormalsVec = std::vector<Vec3>();
    auto faceAreasVec = std::vector<scalar>();
    auto faceUnitNormalsVec = std::vector<Vec3>();
    auto deltaVec = std::vector<Vec3>();
    auto weightsVec = std::vector<scalar>();
    auto deltaCoeffsVec = std::vector<scalar>();
    faceOwnersVec.reserve(nBoundaryFaces);
    faceCentersVec.reserve(nBoundaryFaces);
    ownerCentersVec.reserve(nBoundaryFaces);
    faceNormalsVec.reserve(nBoundaryFaces);
    faceAreasVec.reserve(nBoundaryFaces);
    faceUnitNormalsVec.reserve(nBoundaryFaces);
    deltaVec.reserve(nBoundaryFaces);
    weightsVec.reserve(nBoundaryFaces);
    deltaCoeffsVec.reserve(nBoundaryFaces);

    for (const auto& patch : ordered)
    {
        const localIdx start = patch.baseIndex * nFacesPerPatch;
        for (localIdx f = 0; f < nFacesPerPatch; ++f)
        {
            const localIdx bi = start + f;
            faceOwnersVec.push_back(bOwners[bi]);
            faceCentersVec.push_back(bCenters[bi]);
            ownerCentersVec.push_back(bOwnerCC[bi]);
            faceNormalsVec.push_back(bNormals[bi]);
            faceAreasVec.push_back(bAreas[bi]);
            faceUnitNormalsVec.push_back(bUnitNormals[bi]);
            deltaVec.push_back(bDelta[bi]);
            weightsVec.push_back(patch.normalSign);
            deltaCoeffsVec.push_back(bDeltaCoeffs[bi]);
        }
    }

    auto offset = std::vector<localIdx> {
        0, nFacesPerPatch, 2 * nFacesPerPatch, 3 * nFacesPerPatch, 4 * nFacesPerPatch
    };

    BoundaryMesh boundaryMesh(
        exec,
        {exec, faceOwnersVec},
        {exec, faceCentersVec},
        {exec, ownerCentersVec},
        {exec, faceNormalsVec},
        {exec, faceAreasVec},
        {exec, faceUnitNormalsVec},
        {exec, deltaVec},
        {exec, weightsVec},
        {exec, deltaCoeffsVec},
        offset,
        nProcPatches,
        neighRanks
    );

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

    // Per-proc-patch face count (in fvBoundaryMesh proc-patch / offset order).
    auto patchSizes = std::vector<int>(neighbourRanks.size(), 0);

    // ACCUMULATE per target rank. A single rank may own more than one proc patch to the SAME
    // neighbour rank (typical on the middle rank of a Y-split / sheared cut); the previous code
    // assigned sendCounts[targetRank] = patchSize, silently overwriting the first patch's count
    // with the second. allToAllV needs the TOTAL count per peer, so accumulate with +=.
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        const auto patchSize = static_cast<int>(
            offsets[static_cast<std::size_t>(nInnerBoundaries + i + 1)]
            - offsets[static_cast<std::size_t>(nInnerBoundaries + i)]
        );
        patchSizes[static_cast<std::size_t>(i)] = patchSize;
        sendCounts[static_cast<std::size_t>(targetRank)] += patchSize;
        sendCounts[static_cast<std::size_t>(mpiEnviron.sizeRank())] += patchSize;
    }

    const auto globalOffset = mesh.globalOffset();
    const auto faceCellsH = mesh.boundaryMesh().faceOwners().copyToHost();

    // Send-displacement = prefix sum of per-rank send counts. This defines a contiguous block per
    // peer in the send buffer, so a peer that receives from two non-adjacent proc patches still
    // gets a single contiguous run (allToAllV cannot express a gapped per-peer region).
    auto sdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int r = 1; r < mpiEnviron.sizeRank(); ++r)
    {
        sdispl[static_cast<std::size_t>(r)] =
            sdispl[static_cast<std::size_t>(r - 1)] + sendCounts[static_cast<std::size_t>(r - 1)];
    }

    // Pack the send buffer GROUPED BY TARGET RANK, honouring sdispl. A per-rank write cursor lets
    // multiple proc patches to the same neighbour land contiguously (in proc-patch order) within
    // that rank's block.
    auto writeCursor = sdispl; // copy: running write position per rank
    std::vector<int> sendBuffer(
        static_cast<std::size_t>(sendCounts[static_cast<std::size_t>(mpiEnviron.sizeRank())])
    );
    const auto faceCellsView = faceCellsH.view();
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank =
            static_cast<std::size_t>(neighbourRanks[static_cast<std::size_t>(i)]);
        const localIdx patchStart = offsets[static_cast<std::size_t>(nInnerBoundaries + i)];
        const int patchSize = patchSizes[static_cast<std::size_t>(i)];
        for (int k = 0; k < patchSize; ++k)
        {
            const auto globalCell = static_cast<int>(
                faceCellsView[patchStart + static_cast<localIdx>(k)] + globalOffset
            );
            sendBuffer[static_cast<std::size_t>(writeCursor[targetRank]++)] = globalCell;
        }
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

    auto recvRankGrouped = std::vector<int>(static_cast<std::size_t>(totalRecv));

    mpi::allToAllV<int>(
        sendBuffer.data(),
        sendCounts.data(),
        sdispl.data(),
        recvRankGrouped.data(),
        recvCounts.data(),
        rdispl.data(),
        mpiEnviron.comm()
    );

    // allToAllV delivers recvRankGrouped ordered by ASCENDING SOURCE RANK (one contiguous block
    // per peer, at rdispl[r]). The off-diagonal sparsity, however, consumes recvIdx in LOCAL
    // proc-FACE order: colIdx[pfacei] = recvIdx[pfacei], where pfacei runs over the local processor
    // faces in proc-patch (offset) order. Those two orderings coincide only when the local
    // proc-patch order equals ascending neighbour-rank order (e.g. a 1D strip, or rank 0/1 of a
    // 2x2 grid). On a rank whose patch order differs (rank 2 of a 2x2 grid has patches to [3, 0],
    // not [0, 3]), feeding the rank-grouped buffer directly swaps the off-diagonal columns between
    // patches, mapping each processor face's coupling to the wrong remote cell. That corrupts the
    // convective/diffusive proc-face off-diagonals (worst on the streamwise momentum component),
    // which in turn corrupts hByA and inflates the pressure RHS on large multi-patch meshes.
    //
    // Scatter into proc-face order: walk the local proc patches in offset order and, for each,
    // copy the next run of its neighbour's recv block (advancing a per-neighbour read cursor so
    // multiple local patches sharing one neighbour consume that block sequentially). Within a
    // shared patch both ranks enumerate the faces in the same physical order (the standard
    // processor-patch ordering guarantee already relied upon elsewhere), so a contiguous run lines
    // up face-by-face.
    auto recvIdx = std::vector<int>(static_cast<std::size_t>(totalRecv));
    auto readCursor = rdispl; // per-source-rank running read position into recvRankGrouped
    int writePos = 0;
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto srcRank = static_cast<std::size_t>(neighbourRanks[static_cast<std::size_t>(i)]);
        const int patchSize = patchSizes[static_cast<std::size_t>(i)];
        for (int k = 0; k < patchSize; ++k)
        {
            recvIdx[static_cast<std::size_t>(writePos++)] =
                recvRankGrouped[static_cast<std::size_t>(readCursor[srcRank]++)];
        }
    }

    // boundaryMapVector[rankGroupedPos] = procFacePos (0-based within the proc-boundary block).
    // This is the inverse permutation of the recvIdx scatter walk above. It maps the k-th entry
    // of the rank-grouped recv buffer (recvRankGrouped layout, rdispl[r] offsets) to the local
    // proc-boundary face index. The unified halo primitive scatters through it:
    // value_[procFaceStart + boundaryMapVector[k]].
    std::vector<localIdx> boundaryMapVector(static_cast<std::size_t>(totalRecv));
    {
        // fresh copy of displacement array — DO NOT reuse readCursor (consumed above)
        auto rcsr = rdispl;
        int wpos = 0;
        for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
        {
            const auto src = static_cast<std::size_t>(neighbourRanks[static_cast<std::size_t>(i)]);
            const int psz = patchSizes[static_cast<std::size_t>(i)];
            for (int k = 0; k < psz; ++k)
            {
                boundaryMapVector[static_cast<std::size_t>(rcsr[src]++)] =
                    static_cast<localIdx>(wpos++);
            }
        }
    }
    return CommunicationPattern {sendCounts, recvIdx, boundaryMapVector, mpiEnviron};
}

const CommunicationPattern& cachedCommunicationPattern(const UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    if (!db.contains("communicationPattern"))
    {
        db.insert(
            std::string("communicationPattern"),
            std::make_shared<CommunicationPattern>(computeCommunicationPattern(mesh))
        );
    }
    return *db.get<std::shared_ptr<CommunicationPattern>>("communicationPattern");
}

} // namespace NeoN
