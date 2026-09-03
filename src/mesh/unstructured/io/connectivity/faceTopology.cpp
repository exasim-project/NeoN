// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/connectivity/faceTopology.hpp"
#include "detail.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>


namespace NeoN::io
{

namespace
{

// Face templates for each VTK element type (local node indices per face)
using FaceTemplate = std::vector<std::vector<int>>;

FaceTemplate tetFaces() { return {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}; }

FaceTemplate hexFaces()
{
    return {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 4, 7, 3}, {1, 2, 6, 5}};
}

FaceTemplate wedgeFaces()
{
    return {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {0, 3, 5, 2}};
}

FaceTemplate pyramidFaces() { return {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {0, 4, 3}}; }


// Canonical face key: sorted node indices
using FaceKey = std::vector<localIdx>;

FaceKey makeFaceKey(const std::vector<localIdx>& faceNodeIds)
{
    FaceKey key(faceNodeIds);
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
    localIdx neighbour {-1};
    std::vector<localIdx> nodes;
};


const FaceTemplate& faceTemplateForType(int vtkCellType)
{
    // VTK cell type constants
    constexpr int VTK_TETRA_TYPE = 10;
    constexpr int VTK_HEXAHEDRON_TYPE = 12;
    constexpr int VTK_WEDGE_TYPE = 13;
    constexpr int VTK_PYRAMID_TYPE = 14;

    static const FaceTemplate tetFacesTmpl = tetFaces();
    static const FaceTemplate hexFacesTmpl = hexFaces();
    static const FaceTemplate wedgeFacesTmpl = wedgeFaces();
    static const FaceTemplate pyramidFacesTmpl = pyramidFaces();
    static const FaceTemplate empty;

    switch (vtkCellType)
    {
    case VTK_TETRA_TYPE:
        return tetFacesTmpl;
    case VTK_HEXAHEDRON_TYPE:
        return hexFacesTmpl;
    case VTK_WEDGE_TYPE:
        return wedgeFacesTmpl;
    case VTK_PYRAMID_TYPE:
        return pyramidFacesTmpl;
    default:
        return empty;
    }
}

using FaceMap = std::unordered_map<FaceKey, FaceData, FaceKeyHash>;

FaceMap deduplicateFaces(
    SegmentedVectorView<localIdx, localIdx> fnView, View<int32_t> ctView, localIdx nCells
)
{
    FaceMap faceMap;

    for (localIdx cellId = 0; cellId < nCells; ++cellId)
    {
        int cellType = static_cast<int>(ctView[cellId]);

        auto [start, end] = fnView.bounds(cellId);
        std::vector<localIdx> cellNodes;
        cellNodes.reserve(static_cast<std::size_t>(end - start));
        for (localIdx n = start; n < end; ++n)
        {
            cellNodes.push_back(fnView.values[n]);
        }

        const auto& faces = faceTemplateForType(cellType);
        if (faces.empty()) continue;

        for (const auto& faceLocalNodes : faces)
        {
            std::vector<localIdx> faceGlobalNodes;
            faceGlobalNodes.reserve(faceLocalNodes.size());
            for (int localNode : faceLocalNodes)
            {
                faceGlobalNodes.push_back(cellNodes[static_cast<std::size_t>(localNode)]);
            }

            FaceKey key = makeFaceKey(faceGlobalNodes);
            auto it = faceMap.find(key);
            if (it == faceMap.end())
            {
                FaceData fd;
                fd.owner = cellId;
                fd.nodes = faceGlobalNodes;
                faceMap[key] = fd;
            }
            else
            {
                it->second.neighbour = cellId;
            }
        }
    }

    return faceMap;
}

std::pair<std::vector<FaceData>, std::vector<FaceData>> partitionAndSortFaces(FaceMap faceMap)
{
    std::vector<FaceData> internalFaces;
    std::vector<FaceData> boundaryFaces;

    for (auto& [key, fd] : faceMap)
    {
        if (fd.neighbour >= 0)
        {
            if (fd.owner > fd.neighbour)
            {
                std::swap(fd.owner, fd.neighbour);
                std::reverse(fd.nodes.begin(), fd.nodes.end());
            }
            internalFaces.push_back(fd);
        }
        else
        {
            boundaryFaces.push_back(fd);
        }
    }

    std::sort(
        internalFaces.begin(),
        internalFaces.end(),
        [](const FaceData& a, const FaceData& b)
        {
            if (a.owner != b.owner) return a.owner < b.owner;
            return a.neighbour < b.neighbour;
        }
    );

    std::sort(
        boundaryFaces.begin(),
        boundaryFaces.end(),
        [](const FaceData& a, const FaceData& b) { return a.owner < b.owner; }
    );

    return {internalFaces, boundaryFaces};
}

struct FaceArrays
{
    std::vector<localIdx> ownerData;
    std::vector<localIdx> neighbourData;
    std::vector<localIdx> faceNodesValues;
    std::vector<localIdx> faceNodesSizes;
    localIdx nInternal;
    localIdx nBoundary;
};

FaceArrays packFaceArrays(
    const std::vector<FaceData>& internalFaces, const std::vector<FaceData>& boundaryFaces
)
{
    localIdx nInternal = static_cast<localIdx>(internalFaces.size());
    localIdx nBoundary = static_cast<localIdx>(boundaryFaces.size());
    localIdx nFaces = nInternal + nBoundary;

    std::vector<localIdx> ownerData(static_cast<std::size_t>(nFaces));
    std::vector<localIdx> neighbourData(static_cast<std::size_t>(nInternal));
    std::vector<localIdx> faceNodesValues;
    std::vector<localIdx> faceNodesSizes;

    for (localIdx i = 0; i < nInternal; ++i)
    {
        auto idx = static_cast<std::size_t>(i);
        ownerData[idx] = internalFaces[idx].owner;
        neighbourData[idx] = internalFaces[idx].neighbour;
        faceNodesSizes.push_back(static_cast<localIdx>(internalFaces[idx].nodes.size()));
        faceNodesValues.insert(
            faceNodesValues.end(), internalFaces[idx].nodes.begin(), internalFaces[idx].nodes.end()
        );
    }

    for (localIdx i = 0; i < nBoundary; ++i)
    {
        auto idx = static_cast<std::size_t>(i);
        ownerData[static_cast<std::size_t>(nInternal + i)] = boundaryFaces[idx].owner;
        faceNodesSizes.push_back(static_cast<localIdx>(boundaryFaces[idx].nodes.size()));
        faceNodesValues.insert(
            faceNodesValues.end(), boundaryFaces[idx].nodes.begin(), boundaryFaces[idx].nodes.end()
        );
    }

    return FaceArrays {
        std::move(ownerData),
        std::move(neighbourData),
        std::move(faceNodesValues),
        std::move(faceNodesSizes),
        nInternal,
        nBoundary
    };
}

} // anonymous namespace


FaceTopology buildFaceTopology(const Executor& exec, const CellConnectivity& connectivity)
{
    // Step 1: bring connectivity to host for the sequential deduplication algorithm
    auto hostCellToNodes = connectivity.cellToNodes.copyToHost();
    auto hostCellTypes = connectivity.cellTypes.copyToHost();

    // Steps 2-4: deduplicate, partition, pack
    auto faceMap =
        deduplicateFaces(hostCellToNodes.view(), hostCellTypes.view(), connectivity.nCells);
    auto [internalFaces, boundaryFaces] = partitionAndSortFaces(std::move(faceMap));
    auto arrays = packFaceArrays(internalFaces, boundaryFaces);

    // Step 5: build NeoN types on SerialExecutor, then copy to exec
    SerialExecutor serial;

    return FaceTopology {
        Vector<localIdx>(serial, arrays.ownerData).copyToExecutor(exec),
        Vector<localIdx>(serial, arrays.neighbourData).copyToExecutor(exec),
        detail::makeSegmentedVector(arrays.faceNodesValues, arrays.faceNodesSizes, exec),
        arrays.nInternal,
        arrays.nBoundary
    };
}


} // namespace NeoN::io
