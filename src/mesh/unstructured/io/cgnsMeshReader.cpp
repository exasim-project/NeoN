// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <vtkCGNSReader.h>
#include <vtkCellIterator.h>
#include <vtkCellType.h>
#include <vtkCompositeDataSet.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>

// clang-format off
#include <vtk_cgns.h>
#include VTK_CGNS(cgnslib.h)
// clang-format on

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


namespace NeoN::io
{

namespace detail
{

// Canonical face key for BC matching (sorted node indices)
using FaceKey = std::vector<localIdx>;

FaceKey makeFaceKey(const std::vector<localIdx>& faceNodes)
{
    FaceKey key(faceNodes);
    std::sort(key.begin(), key.end());
    return key;
}

struct FaceKeyHash
{
    std::size_t operator()(const FaceKey& key) const
    {
        std::size_t seed = key.size();
        for (auto& v : key)
        {
            seed ^= static_cast<std::size_t>(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct FaceData
{
    localIdx owner {-1};
    std::vector<localIdx> nodes;
};


// Boundary patch info from CGNS
struct CgnsPatchInfo
{
    std::string name;
    // Element range for this BC (1-based CGNS indices into boundary element section)
    cgsize_t rangeStart {};
    cgsize_t rangeEnd {};
};

struct BoundaryElement
{
    cgsize_t globalElementId {};   // 1-based CGNS global element index
    std::vector<localIdx> nodeIds; // 0-based node indices
};

struct CgnsBCData
{
    std::vector<CgnsPatchInfo> patches;
    std::vector<BoundaryElement> boundaryElements;
};


CgnsBCData readCgnsBoundaryInfo(const std::string& filePath)
{
    CgnsBCData result;

    int fn = 0;
    if (cg_open(filePath.c_str(), CG_MODE_READ, &fn) != CG_OK)
    {
        return result;
    }

    int nBases = 0;
    cg_nbases(fn, &nBases);
    if (nBases < 1)
    {
        cg_close(fn);
        return result;
    }

    // Read first base, first zone
    int cellDim = 0;
    int physDim = 0;
    char baseName[33] = {};
    cg_base_read(fn, 1, baseName, &cellDim, &physDim);

    int nZones = 0;
    cg_nzones(fn, 1, &nZones);
    if (nZones < 1)
    {
        cg_close(fn);
        return result;
    }

    // Read zone sizes
    CGNS_ENUMT(ZoneType_t) zoneType;
    cg_zone_type(fn, 1, 1, &zoneType);

    cgsize_t sizes[3] = {};
    char zoneName[33] = {};
    cg_zone_read(fn, 1, 1, zoneName, sizes);
    cgsize_t nVertices = sizes[0];
    cgsize_t nElements = sizes[1];

    // Read element sections to find boundary elements
    int nSections = 0;
    cg_nsections(fn, 1, 1, &nSections);

    // Find boundary element sections (TRI_3, QUAD_4 for 3D meshes)
    struct ElemSection
    {
        int sectionIdx;
        cgsize_t start;
        cgsize_t end;
        CGNS_ENUMT(ElementType_t) type;
    };
    std::vector<ElemSection> boundarySections;

    for (int s = 1; s <= nSections; ++s)
    {
        char secName[33] = {};
        CGNS_ENUMT(ElementType_t) elemType;
        cgsize_t start = 0;
        cgsize_t end = 0;
        int nBnd = 0;
        int parentFlag = 0;
        cg_section_read(fn, 1, 1, s, secName, &elemType, &start, &end, &nBnd, &parentFlag);

        // For 3D meshes, boundary faces are TRI_3 or QUAD_4
        if (cellDim == 3 && (elemType == CGNS_ENUMV(TRI_3) || elemType == CGNS_ENUMV(QUAD_4)))
        {
            boundarySections.push_back({s, start, end, elemType});
        }
        // For 2D meshes, boundary edges are BAR_2
        if (cellDim == 2 && elemType == CGNS_ENUMV(BAR_2))
        {
            boundarySections.push_back({s, start, end, elemType});
        }
    }

    // Read boundary face connectivity for each boundary section
    for (const auto& sec : boundarySections)
    {
        cgsize_t nElems = sec.end - sec.start + 1;
        int nodesPerElem = 0;
        if (sec.type == CGNS_ENUMV(TRI_3)) nodesPerElem = 3;
        else if (sec.type == CGNS_ENUMV(QUAD_4))
            nodesPerElem = 4;
        else if (sec.type == CGNS_ENUMV(BAR_2))
            nodesPerElem = 2;

        std::vector<cgsize_t> connectivity(static_cast<std::size_t>(nElems * nodesPerElem));
        cgsize_t parentData = 0;
        cg_elements_read(fn, 1, 1, sec.sectionIdx, connectivity.data(), &parentData);

        for (cgsize_t e = 0; e < nElems; ++e)
        {
            BoundaryElement be;
            be.globalElementId = sec.start + e; // 1-based global element index
            for (int n = 0; n < nodesPerElem; ++n)
            {
                // CGNS uses 1-based indexing for nodes
                cgsize_t nodeId = connectivity[static_cast<std::size_t>(e * nodesPerElem + n)];
                be.nodeIds.push_back(static_cast<localIdx>(nodeId - 1));
            }
            result.boundaryElements.push_back(be);
        }
    }

    // Read BCs
    int nBCs = 0;
    cg_nbocos(fn, 1, 1, &nBCs);

    for (int bc = 1; bc <= nBCs; ++bc)
    {
        char bcName[33] = {};
        CGNS_ENUMT(BCType_t) bcType;
        CGNS_ENUMT(PointSetType_t) ptsetType;
        cgsize_t npnts = 0;
        int normalIndex[3] = {};
        cgsize_t normalListSize = 0;
        CGNS_ENUMT(DataType_t) normalDataType;
        int ndataset = 0;

        cg_boco_info(
            fn,
            1,
            1,
            bc,
            bcName,
            &bcType,
            &ptsetType,
            &npnts,
            normalIndex,
            &normalListSize,
            &normalDataType,
            &ndataset
        );

        CgnsPatchInfo patch;
        patch.name = std::string(bcName);

        if (ptsetType == CGNS_ENUMV(PointRange) && npnts == 2)
        {
            cgsize_t pnts[2] = {};
            cg_boco_read(fn, 1, 1, bc, pnts, nullptr);
            patch.rangeStart = pnts[0];
            patch.rangeEnd = pnts[1];
        }
        else if (ptsetType == CGNS_ENUMV(ElementRange) && npnts == 2)
        {
            cgsize_t pnts[2] = {};
            cg_boco_read(fn, 1, 1, bc, pnts, nullptr);
            patch.rangeStart = pnts[0];
            patch.rangeEnd = pnts[1];
        }
        else if (ptsetType == CGNS_ENUMV(PointList))
        {
            std::vector<cgsize_t> pnts(static_cast<std::size_t>(npnts));
            cg_boco_read(fn, 1, 1, bc, pnts.data(), nullptr);
            if (npnts > 0)
            {
                patch.rangeStart = pnts[0];
                patch.rangeEnd = pnts[static_cast<std::size_t>(npnts - 1)];
            }
        }

        // Only include BCs whose range overlaps with boundary element sections
        bool overlapsBoundary = false;
        for (const auto& sec : boundarySections)
        {
            if (patch.rangeStart <= sec.end && patch.rangeEnd >= sec.start)
            {
                overlapsBoundary = true;
                break;
            }
        }
        if (overlapsBoundary)
        {
            result.patches.push_back(patch);
        }
    }

    cg_close(fn);
    return result;
}

} // namespace detail


UnstructuredMesh readCgns(const std::string& filePath, const Executor& exec)
{
    // Read CGNS file using VTK for grid data
    vtkNew<vtkCGNSReader> reader;
    reader->SetFileName(filePath.c_str());
    reader->EnableAllBases();
    reader->Update();

    auto* output = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
    if (!output || output->GetNumberOfBlocks() == 0)
    {
        throw std::runtime_error("Failed to read CGNS file: " + filePath);
    }

    // Navigate block hierarchy to find vtkUnstructuredGrid
    vtkUnstructuredGrid* grid = nullptr;
    std::function<vtkUnstructuredGrid*(vtkDataObject*)> findGrid;
    findGrid = [&](vtkDataObject* obj) -> vtkUnstructuredGrid*
    {
        if (auto* g = vtkUnstructuredGrid::SafeDownCast(obj))
        {
            return g;
        }
        if (auto* mb = vtkMultiBlockDataSet::SafeDownCast(obj))
        {
            for (unsigned int i = 0; i < mb->GetNumberOfBlocks(); ++i)
            {
                if (auto* result = findGrid(mb->GetBlock(i)))
                {
                    return result;
                }
            }
        }
        return nullptr;
    };
    grid = findGrid(output);

    if (!grid)
    {
        throw std::runtime_error("No vtkUnstructuredGrid found in CGNS file: " + filePath);
    }

    // Extract points
    localIdx nPoints = static_cast<localIdx>(grid->GetNumberOfPoints());
    std::vector<Vec3> hostPoints(static_cast<std::size_t>(nPoints));
    for (localIdx i = 0; i < nPoints; ++i)
    {
        double p[3];
        grid->GetPoint(i, p);
        hostPoints[static_cast<std::size_t>(i)] =
            Vec3 {static_cast<scalar>(p[0]), static_cast<scalar>(p[1]), static_cast<scalar>(p[2])};
    }

    // Extract cell connectivity and build face topology via meshConverter
    auto conn = extractCellConnectivity(grid);
    localIdx nCells = conn.nCells;
    auto topo = buildFaceTopology(conn);

    localIdx nInternalFaces = topo.nInternalFaces;
    localIdx nBoundaryFaces = topo.nBoundaryFaces;
    localIdx nFaces = nInternalFaces + nBoundaryFaces;

    // Read BC info directly from CGNS file
    auto bcData = detail::readCgnsBoundaryInfo(filePath);

    localIdx nBoundaries = 0;
    std::vector<localIdx> patchOffsets;

    if (!bcData.patches.empty() && !bcData.boundaryElements.empty())
    {
        // Build a mapping from boundary face canonical key → boundary face index (0-based)
        std::unordered_map<detail::FaceKey, localIdx, detail::FaceKeyHash> bndFaceMap;
        for (localIdx f = nInternalFaces; f < nFaces; ++f)
        {
            auto fi = static_cast<std::size_t>(f);
            detail::FaceKey key = detail::makeFaceKey(topo.faceNodes[fi]);
            bndFaceMap[key] = f - nInternalFaces;
        }

        // For each patch, collect the boundary face indices by matching
        // the BC PointRange (element IDs) to the boundary elements we read
        std::vector<std::vector<localIdx>> patchFaceIndices(bcData.patches.size());

        for (std::size_t p = 0; p < bcData.patches.size(); ++p)
        {
            const auto& patch = bcData.patches[p];
            for (const auto& be : bcData.boundaryElements)
            {
                if (be.globalElementId >= patch.rangeStart && be.globalElementId <= patch.rangeEnd)
                {
                    detail::FaceKey key = detail::makeFaceKey(be.nodeIds);
                    auto it = bndFaceMap.find(key);
                    if (it != bndFaceMap.end())
                    {
                        patchFaceIndices[p].push_back(it->second);
                    }
                }
            }
        }

        // Reorder boundary faces by patch
        nBoundaries = static_cast<localIdx>(bcData.patches.size());
        patchOffsets.push_back(0);

        std::vector<detail::FaceData> reorderedBndFaces;
        reorderedBndFaces.reserve(static_cast<std::size_t>(nBoundaryFaces));
        std::vector<bool> matched(static_cast<std::size_t>(nBoundaryFaces), false);

        for (std::size_t p = 0; p < bcData.patches.size(); ++p)
        {
            for (localIdx idx : patchFaceIndices[p])
            {
                auto fi = static_cast<std::size_t>(nInternalFaces + idx);
                detail::FaceData fd;
                fd.owner = topo.faceOwner[fi];
                fd.nodes = topo.faceNodes[fi];
                reorderedBndFaces.push_back(fd);
                matched[static_cast<std::size_t>(idx)] = true;
            }
            patchOffsets.push_back(static_cast<localIdx>(reorderedBndFaces.size()));
        }

        // Add unmatched boundary faces as an extra "default" patch
        bool hasUnmatched = false;
        for (localIdx i = 0; i < nBoundaryFaces; ++i)
        {
            if (!matched[static_cast<std::size_t>(i)])
            {
                hasUnmatched = true;
                auto fi = static_cast<std::size_t>(nInternalFaces + i);
                detail::FaceData fd;
                fd.owner = topo.faceOwner[fi];
                fd.nodes = topo.faceNodes[fi];
                reorderedBndFaces.push_back(fd);
            }
        }
        if (hasUnmatched)
        {
            ++nBoundaries;
            patchOffsets.push_back(static_cast<localIdx>(reorderedBndFaces.size()));
        }

        // Update topology with reordered boundary faces
        for (localIdx i = 0; i < static_cast<localIdx>(reorderedBndFaces.size()); ++i)
        {
            auto fi = static_cast<std::size_t>(nInternalFaces + i);
            topo.faceOwner[fi] = reorderedBndFaces[static_cast<std::size_t>(i)].owner;
            topo.faceNodes[fi] = reorderedBndFaces[static_cast<std::size_t>(i)].nodes;
        }
    }
    else
    {
        // No BC info available: single patch with all boundary faces
        nBoundaries = (nBoundaryFaces > 0) ? 1 : 0;
        patchOffsets = {0, nBoundaryFaces};
    }

    // Compute geometry (after potential face reordering)
    auto geom = computeGeometry(hostPoints, topo, nCells);

    // Get host views for boundary mesh construction
    auto hostGeom = MeshGeometry {
        geom.cellVolumes.copyToHost(),
        geom.cellCentres.copyToHost(),
        geom.faceAreas.copyToHost(),
        geom.faceCentres.copyToHost(),
        geom.magFaceAreas.copyToHost()
    };
    auto hFaceCentres = hostGeom.faceCentres.view();
    auto hCellCentres = hostGeom.cellCentres.view();
    auto hFaceAreas = hostGeom.faceAreas.view();
    auto hMagFaceAreas = hostGeom.magFaceAreas.view();

    // Build NeoN vectors on the target executor
    vectorVector meshPoints(exec, hostPoints);
    auto cellVolumes = geom.cellVolumes.copyToExecutor(exec);
    auto cellCentres = geom.cellCentres.copyToExecutor(exec);
    auto faceAreasVec = geom.faceAreas.copyToExecutor(exec);
    auto faceCentresVec = geom.faceCentres.copyToExecutor(exec);
    auto magFaceAreasVec = geom.magFaceAreas.copyToExecutor(exec);
    labelVector faceOwnerVec(exec, topo.faceOwner);

    // faceNeighbour only covers internal faces
    labelVector faceNeighbourVec(exec, topo.faceNeighbour);

    // Build BoundaryMesh
    std::vector<label> bndFaceCells(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndCf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndCn(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndSf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndMagSf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndNf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndDelta(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndWeights(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndDeltaCoeffs(static_cast<std::size_t>(nBoundaryFaces));

    for (localIdx i = 0; i < nBoundaryFaces; ++i)
    {
        auto bi = static_cast<std::size_t>(i);
        localIdx fi = nInternalFaces + i;

        localIdx ownerCell = topo.faceOwner[static_cast<std::size_t>(fi)];

        bndFaceCells[bi] = static_cast<label>(ownerCell);
        bndCf[bi] = hFaceCentres[fi];
        bndCn[bi] = hCellCentres[ownerCell];
        bndSf[bi] = hFaceAreas[fi];
        bndMagSf[bi] = hMagFaceAreas[fi];

        // Unit normal
        if (bndMagSf[bi] > 1e-30)
        {
            bndNf[bi] = bndSf[bi] * (1.0 / bndMagSf[bi]);
        }
        else
        {
            bndNf[bi] = Vec3 {0.0, 0.0, 0.0};
        }

        // Delta = face centre - cell centre
        bndDelta[bi] = bndCf[bi] - bndCn[bi];

        // Delta coefficient = 1 / |delta|
        scalar magDelta = mag(bndDelta[bi]);
        bndDeltaCoeffs[bi] = (magDelta > 1e-30) ? 1.0 / magDelta : 0.0;

        // Weight = 1 for boundary faces (full weight to internal cell)
        bndWeights[bi] = 1.0;
    }

    BoundaryMesh boundaryMesh(
        exec,
        labelVector(exec, bndFaceCells),
        vectorVector(exec, bndCf),
        vectorVector(exec, bndCn),
        vectorVector(exec, bndSf),
        scalarVector(exec, bndMagSf),
        vectorVector(exec, bndNf),
        vectorVector(exec, bndDelta),
        scalarVector(exec, bndWeights),
        scalarVector(exec, bndDeltaCoeffs),
        patchOffsets
    );

    auto mesh = UnstructuredMesh(
        exec,
        meshPoints,
        cellVolumes,
        cellCentres,
        faceAreasVec,
        faceCentresVec,
        magFaceAreasVec,
        faceOwnerVec,
        faceNeighbourVec,
        nCells,
        nInternalFaces,
        nBoundaryFaces,
        nBoundaries,
        nFaces,
        boundaryMesh
    );

    // Store face node connectivity in stencilDB for writer round-trip
    auto faceNodePtr = std::make_shared<std::vector<std::vector<localIdx>>>(topo.faceNodes);
    mesh.stencilDB().insert(std::string("io::faceNodes"), faceNodePtr);

    // Store patch names in stencilDB
    if (!bcData.patches.empty())
    {
        auto patchNames = std::make_shared<std::vector<std::string>>();
        for (const auto& patch : bcData.patches)
        {
            patchNames->push_back(patch.name);
        }
        mesh.stencilDB().insert(std::string("io::patchNames"), patchNames);
    }

    return mesh;
}


} // namespace NeoN::io
