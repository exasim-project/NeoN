// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/partition/extractSubMesh.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace NeoN::partition
{

namespace
{

/// A proc-boundary face created from a global internal face.
struct ProcFace
{
    localIdx globalFaceIdx; ///< original global face index
    localIdx subOwner;      ///< local cell index (in partition) that owns this face
    localIdx globalOther;   ///< global cell index on the other side (outside partition)
    bool flipped;           ///< true if face normal must be negated (owner was outside)
};

} // namespace

UnstructuredMesh
extractSubMesh(const UnstructuredMesh& mesh, const std::vector<int>& cellPart, int partId)
{
    const Executor exec = mesh.exec();
    const localIdx nCells = mesh.nCells();
    const localIdx nInternal = mesh.nInternalFaces();
    const localIdx nFaces = mesh.nFaces();
    const localIdx nBoundaries = mesh.nBoundaries();

    // --- Copy all mesh arrays to host ---
    auto hOwner = mesh.faceOwner().copyToHost();
    auto hNeighbour = mesh.faceNeighbour().copyToHost();
    auto hPoints = mesh.points().copyToHost();
    auto hCellVol = mesh.cellVolumes().copyToHost();
    auto hCellCentres = mesh.cellCentres().copyToHost();
    auto hFaceAreas = mesh.faceAreas().copyToHost();
    auto hFaceCentres = mesh.faceCentres().copyToHost();
    auto hMagFA = mesh.magFaceAreas().copyToHost();

    auto ownerV = hOwner.view();
    auto neighbourV = hNeighbour.view();
    auto pointsV = hPoints.view();
    auto cellVolV = hCellVol.view();
    auto cellCentV = hCellCentres.view();
    auto faceAreasV = hFaceAreas.view();
    auto faceCentV = hFaceCentres.view();
    auto magFAV = hMagFA.view();

    // --- Boundary mesh host data ---
    auto hBndFaceCells = mesh.boundaryMesh().faceCells().copyToHost();
    auto hBndCf = mesh.boundaryMesh().cf().copyToHost();
    auto hBndCn = mesh.boundaryMesh().cn().copyToHost();
    auto hBndSf = mesh.boundaryMesh().sf().copyToHost();
    auto hBndMagSf = mesh.boundaryMesh().magSf().copyToHost();
    auto hBndNf = mesh.boundaryMesh().nf().copyToHost();
    auto hBndDelta = mesh.boundaryMesh().delta().copyToHost();
    auto hBndWeights = mesh.boundaryMesh().weights().copyToHost();
    auto hBndDeltaCoeffs = mesh.boundaryMesh().deltaCoeffs().copyToHost();
    const std::vector<localIdx>& bndOffset = mesh.boundaryMesh().offset();

    // --- stencilDB keys ---
    auto& globalFaceNodes =
        *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>("io::faceNodes");
    const auto& globalPatchNames =
        *mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");

    // -----------------------------------------------------------------------
    // Step 1: Collect local cells and build globalToLocalCell map
    // -----------------------------------------------------------------------
    std::vector<localIdx> localCells;
    localCells.reserve(static_cast<std::size_t>(nCells / 4));
    for (localIdx c = 0; c < nCells; ++c)
    {
        if (cellPart[static_cast<std::size_t>(c)] == partId) localCells.push_back(c);
    }
    const localIdx nSubCells = static_cast<localIdx>(localCells.size());

    std::unordered_map<localIdx, localIdx> g2lCell;
    g2lCell.reserve(static_cast<std::size_t>(nSubCells));
    for (localIdx lc = 0; lc < nSubCells; ++lc)
        g2lCell[localCells[static_cast<std::size_t>(lc)]] = lc;

    auto inPart = [&](label c) -> bool { return g2lCell.count(static_cast<localIdx>(c)) > 0; };

    // -----------------------------------------------------------------------
    // Step 2: Face classification
    // -----------------------------------------------------------------------
    std::vector<localIdx> subInternal; // global face idx for sub-internal faces
    std::vector<ProcFace> procFaces;   // inter-partition faces

    // Per original patch: global boundary face indices (in boundary-face numbering)
    // bndFaceIdx = f - nInternal  (0-based boundary face index)
    std::vector<std::vector<localIdx>> patchBndFaces(static_cast<std::size_t>(nBoundaries)
    ); // global boundary-face indices (0-based within boundary array)

    for (localIdx f = 0; f < nInternal; ++f)
    {
        bool ownIn = inPart(ownerV[f]);
        bool nbIn = inPart(neighbourV[f]);
        if (ownIn && nbIn)
        {
            subInternal.push_back(f);
        }
        else if (ownIn && !nbIn)
        {
            procFaces.push_back(
                {f,
                 g2lCell.at(static_cast<localIdx>(ownerV[f])),
                 static_cast<localIdx>(neighbourV[f]),
                 false}
            );
        }
        else if (!ownIn && nbIn)
        {
            procFaces.push_back(
                {f,
                 g2lCell.at(static_cast<localIdx>(neighbourV[f])),
                 static_cast<localIdx>(ownerV[f]),
                 true}
            );
        }
    }

    // Classify original boundary faces
    for (localIdx b = 0; b < nBoundaries; ++b)
    {
        for (localIdx bi = bndOffset[static_cast<std::size_t>(b)];
             bi < bndOffset[static_cast<std::size_t>(b) + 1];
             ++bi)
        {
            label c = hBndFaceCells.view()[bi];
            if (inPart(c))
                patchBndFaces[static_cast<std::size_t>(b)].push_back(static_cast<localIdx>(bi));
        }
    }

    // -----------------------------------------------------------------------
    // Step 2b: Collect unique ghost cells from proc-boundary faces
    // -----------------------------------------------------------------------
    std::unordered_map<localIdx, localIdx> ghostG2L; // global -> ghost-local index
    std::vector<localIdx> ghostCellGlobalIds;

    for (const auto& pf : procFaces)
    {
        if (ghostG2L.count(pf.globalOther) == 0)
        {
            ghostG2L[pf.globalOther] = static_cast<localIdx>(ghostCellGlobalIds.size());
            ghostCellGlobalIds.push_back(pf.globalOther);
        }
    }

    const localIdx nGhostCells = static_cast<localIdx>(ghostCellGlobalIds.size());

    std::vector<scalar> ghostCellVols(static_cast<std::size_t>(nGhostCells));
    std::vector<Vec3> ghostCellCentres(static_cast<std::size_t>(nGhostCells));
    for (localIdx gi = 0; gi < nGhostCells; ++gi)
    {
        localIdx gc = ghostCellGlobalIds[static_cast<std::size_t>(gi)];
        ghostCellVols[static_cast<std::size_t>(gi)] = cellVolV[gc];
        ghostCellCentres[static_cast<std::size_t>(gi)] = cellCentV[gc];
    }

    // -----------------------------------------------------------------------
    // Step 3: Point collection — walk all sub-faces, collect unique point indices
    // -----------------------------------------------------------------------
    std::unordered_map<localIdx, localIdx> g2lPoint;
    std::vector<localIdx> localPoints;

    auto addPoint = [&](localIdx gp)
    {
        if (g2lPoint.count(gp) == 0)
        {
            g2lPoint[gp] = static_cast<localIdx>(localPoints.size());
            localPoints.push_back(gp);
        }
    };

    for (localIdx gf : subInternal)
        for (localIdx gp : globalFaceNodes[static_cast<std::size_t>(gf)])
            addPoint(gp);

    for (auto& pf : procFaces)
        for (localIdx gp : globalFaceNodes[static_cast<std::size_t>(pf.globalFaceIdx)])
            addPoint(gp);

    for (localIdx b = 0; b < nBoundaries; ++b)
    {
        localIdx globalFaceStart = nInternal + bndOffset[static_cast<std::size_t>(b)];
        for (localIdx bi : patchBndFaces[static_cast<std::size_t>(b)])
        {
            localIdx gf = globalFaceStart + bi - bndOffset[static_cast<std::size_t>(b)];
            for (localIdx gp : globalFaceNodes[static_cast<std::size_t>(gf)])
                addPoint(gp);
        }
    }

    // Record the sub-mesh point count before adding ghost-only points
    const localIdx nSubPointsBeforeGhosts = static_cast<localIdx>(localPoints.size());

    // -----------------------------------------------------------------------
    // Step 3b: Ghost cell face-node collection
    // -----------------------------------------------------------------------
    // For each ghost cell, find all its faces in the global mesh and collect
    // their node indices (remapped to local point numbering, extending as needed).
    std::vector<std::vector<std::vector<localIdx>>> ghostCellFaceNodes(
        static_cast<std::size_t>(nGhostCells)
    );

    if (nGhostCells > 0)
    {
        // Walk all global internal faces to find faces owned by ghost cells
        for (localIdx f = 0; f < nInternal; ++f)
        {
            auto o = static_cast<localIdx>(ownerV[f]);
            auto n = static_cast<localIdx>(neighbourV[f]);

            auto itO = ghostG2L.find(o);
            auto itN = ghostG2L.find(n);

            if (itO != ghostG2L.end())
            {
                auto& gNodes = globalFaceNodes[static_cast<std::size_t>(f)];
                std::vector<localIdx> mapped;
                mapped.reserve(gNodes.size());
                for (localIdx gp : gNodes)
                {
                    addPoint(gp);
                    mapped.push_back(g2lPoint.at(gp));
                }
                ghostCellFaceNodes[static_cast<std::size_t>(itO->second)].push_back(mapped);
            }
            if (itN != ghostG2L.end())
            {
                auto& gNodes = globalFaceNodes[static_cast<std::size_t>(f)];
                std::vector<localIdx> mapped;
                mapped.reserve(gNodes.size());
                for (localIdx gp : gNodes)
                {
                    addPoint(gp);
                    mapped.push_back(g2lPoint.at(gp));
                }
                ghostCellFaceNodes[static_cast<std::size_t>(itN->second)].push_back(mapped);
            }
        }
        // Walk boundary faces for ghost cells
        for (localIdx b = 0; b < nBoundaries; ++b)
        {
            for (localIdx bi = bndOffset[static_cast<std::size_t>(b)];
                 bi < bndOffset[static_cast<std::size_t>(b) + 1];
                 ++bi)
            {
                auto c = static_cast<localIdx>(hBndFaceCells.view()[bi]);
                auto it = ghostG2L.find(c);
                if (it != ghostG2L.end())
                {
                    localIdx gf = nInternal + bi;
                    auto& gNodes = globalFaceNodes[static_cast<std::size_t>(gf)];
                    std::vector<localIdx> mapped;
                    mapped.reserve(gNodes.size());
                    for (localIdx gp : gNodes)
                    {
                        addPoint(gp);
                        mapped.push_back(g2lPoint.at(gp));
                    }
                    ghostCellFaceNodes[static_cast<std::size_t>(it->second)].push_back(mapped);
                }
            }
        }
    }

    // Ghost-only points (added beyond nSubPointsBeforeGhosts)
    std::vector<Vec3> ghostPoints;
    for (localIdx lp = nSubPointsBeforeGhosts; lp < static_cast<localIdx>(localPoints.size()); ++lp)
    {
        ghostPoints.push_back(pointsV[localPoints[static_cast<std::size_t>(lp)]]);
    }

    // -----------------------------------------------------------------------
    // Step 4: Build sub-mesh face arrays
    // -----------------------------------------------------------------------
    const localIdx nSubInternal = static_cast<localIdx>(subInternal.size());
    // Count sub boundary faces
    localIdx nSubOrigBnd = 0;
    for (localIdx b = 0; b < nBoundaries; ++b)
        nSubOrigBnd += static_cast<localIdx>(patchBndFaces[static_cast<std::size_t>(b)].size());

    const localIdx nProcFaces = static_cast<localIdx>(procFaces.size());
    const localIdx nSubBnd = nSubOrigBnd + nProcFaces;
    const localIdx nSubFaces = nSubInternal + nSubBnd;
    const localIdx nSubPoints = nSubPointsBeforeGhosts;

    // Sub-face arrays
    std::vector<Vec3> subFaceAreas(static_cast<std::size_t>(nSubFaces));
    std::vector<Vec3> subFaceCentres(static_cast<std::size_t>(nSubFaces));
    std::vector<scalar> subMagFA(static_cast<std::size_t>(nSubFaces));
    std::vector<label> subFaceOwner(static_cast<std::size_t>(nSubFaces));
    std::vector<label> subFaceNeighbour(static_cast<std::size_t>(nSubInternal));

    // Fill sub-internal faces
    for (localIdx sf = 0; sf < nSubInternal; ++sf)
    {
        localIdx gf = subInternal[static_cast<std::size_t>(sf)];
        subFaceAreas[static_cast<std::size_t>(sf)] = faceAreasV[gf];
        subFaceCentres[static_cast<std::size_t>(sf)] = faceCentV[gf];
        subMagFA[static_cast<std::size_t>(sf)] = magFAV[gf];
        subFaceOwner[static_cast<std::size_t>(sf)] =
            static_cast<label>(g2lCell.at(static_cast<localIdx>(ownerV[gf])));
        subFaceNeighbour[static_cast<std::size_t>(sf)] =
            static_cast<label>(g2lCell.at(static_cast<localIdx>(neighbourV[gf])));
    }

    // Boundary data arrays
    std::vector<label> bndFaceCells;
    std::vector<Vec3> bndCf, bndCn, bndSf, bndNf, bndDelta;
    std::vector<scalar> bndMagSf, bndWeights, bndDeltaCoeffs;
    bndFaceCells.reserve(static_cast<std::size_t>(nSubBnd));
    bndCf.reserve(static_cast<std::size_t>(nSubBnd));
    bndCn.reserve(static_cast<std::size_t>(nSubBnd));
    bndSf.reserve(static_cast<std::size_t>(nSubBnd));
    bndNf.reserve(static_cast<std::size_t>(nSubBnd));
    bndDelta.reserve(static_cast<std::size_t>(nSubBnd));
    bndMagSf.reserve(static_cast<std::size_t>(nSubBnd));
    bndWeights.reserve(static_cast<std::size_t>(nSubBnd));
    bndDeltaCoeffs.reserve(static_cast<std::size_t>(nSubBnd));

    // Sub-face node connectivity
    auto subFaceNodesPtr =
        std::make_shared<std::vector<std::vector<localIdx>>>(static_cast<std::size_t>(nSubFaces));
    auto& subFaceNodes = *subFaceNodesPtr;

    // --- Fill internal face nodes ---
    for (localIdx sf = 0; sf < nSubInternal; ++sf)
    {
        localIdx gf = subInternal[static_cast<std::size_t>(sf)];
        auto& gNodes = globalFaceNodes[static_cast<std::size_t>(gf)];
        auto& sNodes = subFaceNodes[static_cast<std::size_t>(sf)];
        sNodes.resize(gNodes.size());
        for (std::size_t k = 0; k < gNodes.size(); ++k)
            sNodes[k] = g2lPoint.at(gNodes[k]);
    }

    // --- Fill original boundary faces per patch ---
    std::vector<localIdx> subBndOffset;
    std::vector<std::string> subPatchNames;
    subBndOffset.push_back(0);

    localIdx subFaceId = nSubInternal; // index into sub-mesh face arrays
    localIdx subBndId = 0;

    for (localIdx b = 0; b < nBoundaries; ++b)
    {
        const auto& biFaces = patchBndFaces[static_cast<std::size_t>(b)];
        if (biFaces.empty()) continue;

        subPatchNames.push_back(globalPatchNames[static_cast<std::size_t>(b)]);

        for (localIdx bi : biFaces)
        {
            // bi is the 0-based index into the boundary face arrays
            // Global face index = nInternal + bi  (since bi runs over all bnd faces globally)
            // Wait: bi is already the 0-based index into bndFaceCells etc.
            // Global face index in faceOwner array = nInternal + bi
            // But faceOwner for boundary faces was stored as: fOwner[nInternal + bndIdx]
            // And bndFaceCells[bi] == fOwner[nInternal + bi] == the adjacent cell

            std::size_t sz = static_cast<std::size_t>(subFaceId);
            subFaceAreas[sz] = hBndSf.view()[bi];
            subFaceCentres[sz] = hBndCf.view()[bi];
            subMagFA[sz] = hBndMagSf.view()[bi];
            label localC =
                static_cast<label>(g2lCell.at(static_cast<localIdx>(hBndFaceCells.view()[bi])));
            subFaceOwner[sz] = localC;

            bndFaceCells.push_back(localC);
            bndCf.push_back(hBndCf.view()[bi]);
            bndCn.push_back(hBndCn.view()[bi]);
            bndSf.push_back(hBndSf.view()[bi]);
            bndMagSf.push_back(hBndMagSf.view()[bi]);
            bndNf.push_back(hBndNf.view()[bi]);
            bndDelta.push_back(hBndDelta.view()[bi]);
            bndWeights.push_back(hBndWeights.view()[bi]);
            bndDeltaCoeffs.push_back(hBndDeltaCoeffs.view()[bi]);

            // Face nodes
            // Global face index for boundary face bi:
            // It's at global face position nInternal + bi
            localIdx gf = nInternal + bi;
            auto& gNodes = globalFaceNodes[static_cast<std::size_t>(gf)];
            auto& sNodes = subFaceNodes[sz];
            sNodes.resize(gNodes.size());
            for (std::size_t k = 0; k < gNodes.size(); ++k)
                sNodes[k] = g2lPoint.at(gNodes[k]);

            ++subFaceId;
            ++subBndId;
        }

        subBndOffset.push_back(static_cast<localIdx>(bndFaceCells.size()));
    }

    // --- Fill proc-boundary faces, grouped by neighbor partition ---
    // Group proc faces by the partition of the neighboring cell
    std::map<int, std::vector<std::size_t>> procFacesByNeighbor;
    for (std::size_t i = 0; i < procFaces.size(); ++i)
    {
        int neighborPart = cellPart[static_cast<std::size_t>(procFaces[i].globalOther)];
        procFacesByNeighbor[neighborPart].push_back(i);
    }

    for (const auto& [neighborPartId, faceIndices] : procFacesByNeighbor)
    {
        subPatchNames.push_back(
            "proc" + std::to_string(partId) + "to" + std::to_string(neighborPartId)
        );

        for (std::size_t idx : faceIndices)
        {
            const auto& pf = procFaces[idx];
            std::size_t sz = static_cast<std::size_t>(subFaceId);
            Vec3 sf = faceAreasV[pf.globalFaceIdx];
            Vec3 fc = faceCentV[pf.globalFaceIdx];
            scalar mgSf = magFAV[pf.globalFaceIdx];
            if (pf.flipped)
            {
                sf = sf * scalar(-1.0);
            }
            Vec3 nf = sf * (scalar(1.0) / mgSf);

            Vec3 ownerCc =
                cellCentV[static_cast<localIdx>(localCells[static_cast<std::size_t>(pf.subOwner)])];
            Vec3 otherCc = cellCentV[pf.globalOther];
            Vec3 delta = fc - ownerCc;
            scalar wt = mag(fc - otherCc) / (mag(fc - ownerCc) + mag(fc - otherCc));
            scalar dc = scalar(1.0) / mag(delta);

            subFaceAreas[sz] = sf;
            subFaceCentres[sz] = fc;
            subMagFA[sz] = mgSf;
            subFaceOwner[sz] = static_cast<label>(pf.subOwner);

            bndFaceCells.push_back(static_cast<label>(pf.subOwner));
            bndCf.push_back(fc);
            bndCn.push_back(otherCc);
            bndSf.push_back(sf);
            bndMagSf.push_back(mgSf);
            bndNf.push_back(nf);
            bndDelta.push_back(delta);
            bndWeights.push_back(wt);
            bndDeltaCoeffs.push_back(dc);

            // Face nodes
            auto& gNodes = globalFaceNodes[static_cast<std::size_t>(pf.globalFaceIdx)];
            auto& sNodes = subFaceNodes[sz];
            sNodes.resize(gNodes.size());
            for (std::size_t k = 0; k < gNodes.size(); ++k)
                sNodes[k] = g2lPoint.at(gNodes[k]);

            ++subFaceId;
            ++subBndId;
        }

        subBndOffset.push_back(static_cast<localIdx>(bndFaceCells.size()));
    }

    // -----------------------------------------------------------------------
    // Step 4b: Build communication maps (sendMap / receiveMap)
    // -----------------------------------------------------------------------
    int nParts = 0;
    for (auto cp : cellPart)
        if (cp >= nParts) nParts = cp + 1;

    auto commSendMapPtr =
        std::make_shared<std::vector<std::vector<localIdx>>>(static_cast<std::size_t>(nParts));
    auto commReceiveMapPtr =
        std::make_shared<std::vector<std::vector<localIdx>>>(static_cast<std::size_t>(nParts));
    auto& commSendMap = *commSendMapPtr;
    auto& commReceiveMap = *commReceiveMapPtr;

    for (const auto& [neighborPartId, faceIndices] : procFacesByNeighbor)
    {
        struct CommTriple
        {
            localIdx sortKey;
            localIdx localIndex;
        };
        std::vector<CommTriple> sendEntries, recvEntries;
        std::unordered_set<localIdx> seenGlobalOther;

        for (std::size_t idx : faceIndices)
        {
            const auto& pf = procFaces[idx];
            if (seenGlobalOther.insert(pf.globalOther).second)
            {
                localIdx globalOwner = localCells[static_cast<std::size_t>(pf.subOwner)];
                sendEntries.push_back({globalOwner, pf.subOwner});

                localIdx ghostLocal = ghostG2L.at(pf.globalOther);
                recvEntries.push_back({pf.globalOther, nSubCells + ghostLocal});
            }
        }

        auto cmp = [](const CommTriple& a, const CommTriple& b) { return a.sortKey < b.sortKey; };
        std::sort(sendEntries.begin(), sendEntries.end(), cmp);
        std::sort(recvEntries.begin(), recvEntries.end(), cmp);

        auto& sendVec = commSendMap[static_cast<std::size_t>(neighborPartId)];
        auto& recvVec = commReceiveMap[static_cast<std::size_t>(neighborPartId)];
        for (const auto& e : sendEntries)
            sendVec.push_back(e.localIndex);
        for (const auto& e : recvEntries)
            recvVec.push_back(e.localIndex);
    }

    // -----------------------------------------------------------------------
    // Step 5: Sub-mesh geometric arrays
    // -----------------------------------------------------------------------
    const localIdx nSubBoundaries = static_cast<localIdx>(subPatchNames.size());

    // Points
    std::vector<Vec3> subPts(static_cast<std::size_t>(nSubPoints));
    for (localIdx lp = 0; lp < nSubPoints; ++lp)
        subPts[static_cast<std::size_t>(lp)] = pointsV[localPoints[static_cast<std::size_t>(lp)]];

    // Cell volumes and centres
    std::vector<scalar> subCellVols(static_cast<std::size_t>(nSubCells));
    std::vector<Vec3> subCellCentres(static_cast<std::size_t>(nSubCells));
    for (localIdx lc = 0; lc < nSubCells; ++lc)
    {
        localIdx gc = localCells[static_cast<std::size_t>(lc)];
        subCellVols[static_cast<std::size_t>(lc)] = cellVolV[gc];
        subCellCentres[static_cast<std::size_t>(lc)] = cellCentV[gc];
    }

    // -----------------------------------------------------------------------
    // Step 6: Assemble UnstructuredMesh
    // -----------------------------------------------------------------------
    BoundaryMesh subBndMesh(
        exec,
        {exec, bndFaceCells},
        {exec, bndCf},
        {exec, bndCn},
        {exec, bndSf},
        {exec, bndMagSf},
        {exec, bndNf},
        {exec, bndDelta},
        {exec, bndWeights},
        {exec, bndDeltaCoeffs},
        subBndOffset
    );

    UnstructuredMesh subMesh(
        exec,
        {exec, subPts},
        {exec, subCellVols},
        {exec, subCellCentres},
        {exec, subFaceAreas},
        {exec, subFaceCentres},
        {exec, subMagFA},
        {exec, subFaceOwner},
        {exec, subFaceNeighbour},
        nSubCells,
        nSubInternal,
        nSubBnd,
        nSubBoundaries,
        nSubFaces,
        std::move(subBndMesh)
    );

    // -----------------------------------------------------------------------
    // Step 7: Populate stencilDB
    // -----------------------------------------------------------------------
    subMesh.stencilDB().insert(std::string("io::faceNodes"), subFaceNodesPtr);
    auto subPatchNamesPtr = std::make_shared<std::vector<std::string>>(subPatchNames);
    subMesh.stencilDB().insert(std::string("io::patchNames"), subPatchNamesPtr);
    subMesh.stencilDB().insert(
        std::string("partition::globalCellIds"), std::make_shared<std::vector<localIdx>>(localCells)
    );
    subMesh.stencilDB().insert(
        std::string("partition::ghostCellGlobalIds"),
        std::make_shared<std::vector<localIdx>>(ghostCellGlobalIds)
    );
    subMesh.stencilDB().insert(
        std::string("partition::ghostCellVolumes"),
        std::make_shared<std::vector<scalar>>(ghostCellVols)
    );
    subMesh.stencilDB().insert(
        std::string("partition::ghostCellCentres"),
        std::make_shared<std::vector<Vec3>>(ghostCellCentres)
    );
    subMesh.stencilDB().insert(
        std::string("partition::ghostCellFaceNodes"),
        std::make_shared<std::vector<std::vector<std::vector<localIdx>>>>(ghostCellFaceNodes)
    );
    subMesh.stencilDB().insert(
        std::string("partition::ghostPoints"), std::make_shared<std::vector<Vec3>>(ghostPoints)
    );
    subMesh.stencilDB().insert(std::string("partition::nParts"), std::make_shared<int>(nParts));
    subMesh.stencilDB().insert(std::string("partition::partId"), std::make_shared<int>(partId));
    subMesh.stencilDB().insert(std::string("partition::commSendMap"), commSendMapPtr);
    subMesh.stencilDB().insert(std::string("partition::commReceiveMap"), commReceiveMapPtr);

    return subMesh;
}

} // namespace NeoN::partition
