// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"

#include <vtkCellIterator.h>
#include <vtkCellType.h>
#include <vtkIdList.h>
#include <vtkUnstructuredGrid.h>

#include <algorithm>
#include <map>
#include <set>
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


} // anonymous namespace


FaceTopology buildFaceTopology(const CellConnectivity& connectivity)
{
    std::unordered_map<FaceKey, FaceData, FaceKeyHash> faceMap;

    for (localIdx cellId = 0; cellId < connectivity.nCells; ++cellId)
    {
        auto ci = static_cast<std::size_t>(cellId);
        int cellType = connectivity.cellTypes[ci];
        const auto& cellNodes = connectivity.cellToNodes[ci];

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

    // Partition into internal and boundary faces
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

    // Sort internal by owner then neighbour
    std::sort(
        internalFaces.begin(),
        internalFaces.end(),
        [](const FaceData& a, const FaceData& b)
        {
            if (a.owner != b.owner) return a.owner < b.owner;
            return a.neighbour < b.neighbour;
        }
    );

    // Sort boundary by owner
    std::sort(
        boundaryFaces.begin(),
        boundaryFaces.end(),
        [](const FaceData& a, const FaceData& b) { return a.owner < b.owner; }
    );

    // Assemble topology
    localIdx nInternal = static_cast<localIdx>(internalFaces.size());
    localIdx nBoundary = static_cast<localIdx>(boundaryFaces.size());
    localIdx nFaces = nInternal + nBoundary;

    FaceTopology topo;
    topo.nInternalFaces = nInternal;
    topo.nBoundaryFaces = nBoundary;
    topo.faceOwner.resize(static_cast<std::size_t>(nFaces));
    topo.faceNeighbour.resize(static_cast<std::size_t>(nInternal));
    topo.faceNodes.resize(static_cast<std::size_t>(nFaces));

    for (localIdx i = 0; i < nInternal; ++i)
    {
        auto idx = static_cast<std::size_t>(i);
        topo.faceOwner[idx] = internalFaces[idx].owner;
        topo.faceNeighbour[idx] = internalFaces[idx].neighbour;
        topo.faceNodes[idx] = internalFaces[idx].nodes;
    }

    for (localIdx i = 0; i < nBoundary; ++i)
    {
        auto idx = static_cast<std::size_t>(i);
        auto faceIdx = static_cast<std::size_t>(nInternal + i);
        topo.faceOwner[faceIdx] = boundaryFaces[idx].owner;
        topo.faceNodes[faceIdx] = boundaryFaces[idx].nodes;
    }

    return topo;
}


CellConnectivity rebuildCellConnectivity(
    const std::vector<label>& faceOwner,
    const std::vector<label>& faceNeighbour,
    const std::vector<std::vector<localIdx>>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
)
{
    // VTK cell type constants
    constexpr int VTK_QUAD_TYPE = 9;
    constexpr int VTK_TETRA_TYPE = 10;
    constexpr int VTK_HEXAHEDRON_TYPE = 12;
    constexpr int VTK_WEDGE_TYPE = 13;
    constexpr int VTK_PYRAMID_TYPE = 14;

    // Collect faces per cell
    std::vector<std::vector<localIdx>> cellFaces(static_cast<std::size_t>(nCells));
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto oi = static_cast<std::size_t>(faceOwner[static_cast<std::size_t>(f)]);
        cellFaces[oi].push_back(f);
        if (f < nInternalFaces)
        {
            auto ni = static_cast<std::size_t>(faceNeighbour[static_cast<std::size_t>(f)]);
            cellFaces[ni].push_back(f);
        }
    }

    CellConnectivity conn;
    conn.nCells = nCells;
    conn.cellToNodes.resize(static_cast<std::size_t>(nCells));
    conn.cellTypes.resize(static_cast<std::size_t>(nCells));

    for (localIdx c = 0; c < nCells; ++c)
    {
        auto ci = static_cast<std::size_t>(c);

        // Collect unique nodes and check max nodes per face
        std::set<localIdx> nodeSet;
        std::size_t maxNodesPerFace = 0;
        for (localIdx f : cellFaces[ci])
        {
            auto fi = static_cast<std::size_t>(f);
            maxNodesPerFace = std::max(maxNodesPerFace, faceNodes[fi].size());
            for (localIdx n : faceNodes[fi])
            {
                nodeSet.insert(n);
            }
        }
        conn.cellToNodes[ci].assign(nodeSet.begin(), nodeSet.end());

        // Determine element type
        localIdx nCellFaces = static_cast<localIdx>(cellFaces[ci].size());
        localIdx nCellNodes = static_cast<localIdx>(nodeSet.size());

        if (maxNodesPerFace <= 2)
        {
            // 2D cell: faces are edges
            conn.cellTypes[ci] = VTK_QUAD_TYPE;
        }
        else if (nCellFaces == 4 && nCellNodes == 4)
            conn.cellTypes[ci] = VTK_TETRA_TYPE;
        else if (nCellFaces == 6 && nCellNodes == 8)
            conn.cellTypes[ci] = VTK_HEXAHEDRON_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 6)
            conn.cellTypes[ci] = VTK_WEDGE_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 5)
            conn.cellTypes[ci] = VTK_PYRAMID_TYPE;
        else if (nCellNodes == 4)
            conn.cellTypes[ci] = VTK_TETRA_TYPE;
        else if (nCellNodes == 8)
            conn.cellTypes[ci] = VTK_HEXAHEDRON_TYPE;
        else
            conn.cellTypes[ci] = VTK_TETRA_TYPE;
    }

    return conn;
}


std::vector<CellInfo> rebuildCellInfo(
    const std::vector<label>& faceOwner,
    const std::vector<label>& faceNeighbour,
    const std::vector<std::vector<localIdx>>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
)
{
    // VTK cell type constants
    constexpr int VTK_QUAD_TYPE = 9;
    constexpr int VTK_TETRA_TYPE = 10;
    constexpr int VTK_HEXAHEDRON_TYPE = 12;
    constexpr int VTK_WEDGE_TYPE = 13;
    constexpr int VTK_PYRAMID_TYPE = 14;

    // Collect faces per cell
    std::vector<std::vector<localIdx>> cellFaces(static_cast<std::size_t>(nCells));
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto oi = static_cast<std::size_t>(faceOwner[static_cast<std::size_t>(f)]);
        cellFaces[oi].push_back(f);
        if (f < nInternalFaces)
        {
            auto ni = static_cast<std::size_t>(faceNeighbour[static_cast<std::size_t>(f)]);
            cellFaces[ni].push_back(f);
        }
    }

    std::vector<CellInfo> cells(static_cast<std::size_t>(nCells));

    for (localIdx c = 0; c < nCells; ++c)
    {
        auto ci = static_cast<std::size_t>(c);
        auto& cell = cells[ci];

        // Collect unique nodes and face info
        std::set<localIdx> nodeSet;
        for (localIdx f : cellFaces[ci])
        {
            auto fi = static_cast<std::size_t>(f);
            cell.cellFaceNodes.push_back(faceNodes[fi]);
            for (localIdx n : faceNodes[fi])
            {
                nodeSet.insert(n);
            }
        }
        cell.nodeIds.assign(nodeSet.begin(), nodeSet.end());

        // Determine element type from number of faces and nodes
        localIdx nCellFaces = static_cast<localIdx>(cell.cellFaceNodes.size());
        localIdx nCellNodes = static_cast<localIdx>(cell.nodeIds.size());

        // Check max nodes per face to distinguish 2D (edges with 2 nodes) from 3D
        std::size_t maxNodesPerFace = 0;
        for (const auto& fn : cell.cellFaceNodes)
        {
            maxNodesPerFace = std::max(maxNodesPerFace, fn.size());
        }

        if (maxNodesPerFace <= 2)
        {
            // 2D cell: faces are edges
            cell.cellType = VTK_QUAD_TYPE;
        }
        else if (nCellFaces == 4 && nCellNodes == 4)
            cell.cellType = VTK_TETRA_TYPE;
        else if (nCellFaces == 6 && nCellNodes == 8)
            cell.cellType = VTK_HEXAHEDRON_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 6)
            cell.cellType = VTK_WEDGE_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 5)
            cell.cellType = VTK_PYRAMID_TYPE;
        else if (nCellNodes == 4)
            cell.cellType = VTK_TETRA_TYPE;
        else if (nCellNodes == 8)
            cell.cellType = VTK_HEXAHEDRON_TYPE;
        else
            cell.cellType = VTK_TETRA_TYPE;
    }

    return cells;
}


std::vector<localIdx> orderQuadNodes(const CellInfo& cell)
{
    // Chain edges to get nodes in connected order around the quad.
    // Each edge (face) has 2 nodes. Start from first edge and follow connectivity.
    auto& edges = cell.cellFaceNodes;
    std::vector<localIdx> ordered;
    ordered.reserve(4);

    // Start with the first edge
    ordered.push_back(edges[0][0]);
    ordered.push_back(edges[0][1]);

    // Build adjacency: for each node, which edges contain it
    std::map<localIdx, std::vector<std::size_t>> nodeToEdges;
    for (std::size_t ei = 0; ei < edges.size(); ++ei)
    {
        for (localIdx n : edges[ei])
        {
            nodeToEdges[n].push_back(ei);
        }
    }

    std::set<std::size_t> usedEdges;
    usedEdges.insert(0);

    // Follow the chain: last node of ordered → find next unused edge containing it
    for (int step = 0; step < 2; ++step)
    {
        localIdx lastNode = ordered.back();
        for (std::size_t ei : nodeToEdges[lastNode])
        {
            if (usedEdges.count(ei)) continue;
            usedEdges.insert(ei);
            // Add the other node of this edge
            localIdx other = (edges[ei][0] == lastNode) ? edges[ei][1] : edges[ei][0];
            ordered.push_back(other);
            break;
        }
    }

    return ordered;
}


std::vector<localIdx> orderTetNodes(const CellInfo& cell)
{
    // Take first face as base (n0, n1, n2), remaining node is apex
    auto& baseFace = cell.cellFaceNodes[0];
    std::set<localIdx> baseNodes(baseFace.begin(), baseFace.end());
    localIdx apex = -1;
    for (localIdx n : cell.nodeIds)
    {
        if (baseNodes.find(n) == baseNodes.end())
        {
            apex = n;
            break;
        }
    }
    return {baseFace[0], baseFace[1], baseFace[2], apex};
}


std::vector<localIdx> orderHexNodes(const CellInfo& cell)
{
    auto& faces = cell.cellFaceNodes;
    auto& bottom = faces[0];
    std::set<localIdx> bottomSet(bottom.begin(), bottom.end());

    // Find top face (shares no nodes with bottom)
    std::size_t topIdx = 1;
    for (std::size_t fi = 1; fi < faces.size(); ++fi)
    {
        bool sharesNode = false;
        for (localIdx n : faces[fi])
        {
            if (bottomSet.count(n))
            {
                sharesNode = true;
                break;
            }
        }
        if (!sharesNode)
        {
            topIdx = fi;
            break;
        }
    }
    std::set<localIdx> topSet(faces[topIdx].begin(), faces[topIdx].end());

    // Build vertical edge map from side faces
    std::map<localIdx, localIdx> bottomToTop;
    for (std::size_t fi = 0; fi < faces.size(); ++fi)
    {
        if (fi == 0 || fi == topIdx) continue;
        auto& sideFace = faces[fi];
        auto sz = sideFace.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            localIdx a = sideFace[i];
            localIdx b = sideFace[(i + 1) % sz];
            bool aBot = bottomSet.count(a) > 0;
            bool bBot = bottomSet.count(b) > 0;
            bool aTop = topSet.count(a) > 0;
            bool bTop = topSet.count(b) > 0;
            if (aBot && bTop) bottomToTop[a] = b;
            else if (bBot && aTop)
                bottomToTop[b] = a;
        }
    }

    std::vector<localIdx> nodes(8);
    for (int i = 0; i < 4; ++i)
    {
        auto bi = static_cast<std::size_t>(i);
        nodes[bi] = bottom[bi];
        auto it = bottomToTop.find(bottom[bi]);
        if (it != bottomToTop.end())
        {
            nodes[bi + 4] = it->second;
        }
    }
    return nodes;
}


std::vector<localIdx> orderPyramidNodes(const CellInfo& cell)
{
    auto& faces = cell.cellFaceNodes;
    std::size_t baseIdx = 0;
    for (std::size_t fi = 0; fi < faces.size(); ++fi)
    {
        if (faces[fi].size() == 4)
        {
            baseIdx = fi;
            break;
        }
    }
    auto& base = faces[baseIdx];
    std::set<localIdx> baseSet(base.begin(), base.end());

    localIdx apex = -1;
    for (localIdx n : cell.nodeIds)
    {
        if (baseSet.find(n) == baseSet.end())
        {
            apex = n;
            break;
        }
    }

    return {base[0], base[1], base[2], base[3], apex};
}


std::vector<localIdx> orderWedgeNodes(const CellInfo& cell)
{
    auto& faces = cell.cellFaceNodes;
    std::vector<std::size_t> triFaces;
    for (std::size_t fi = 0; fi < faces.size(); ++fi)
    {
        if (faces[fi].size() == 3)
        {
            triFaces.push_back(fi);
        }
    }

    auto& bottom = faces[triFaces[0]];
    std::set<localIdx> bottomSet(bottom.begin(), bottom.end());
    std::set<localIdx> topSet(faces[triFaces[1]].begin(), faces[triFaces[1]].end());

    // Build vertical edge map from quad side faces
    std::map<localIdx, localIdx> bottomToTop;
    for (std::size_t fi = 0; fi < faces.size(); ++fi)
    {
        if (faces[fi].size() != 4) continue;
        auto& sideFace = faces[fi];
        auto sz = sideFace.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            localIdx a = sideFace[i];
            localIdx b = sideFace[(i + 1) % sz];
            bool aBot = bottomSet.count(a) > 0;
            bool bBot = bottomSet.count(b) > 0;
            bool aTop = topSet.count(a) > 0;
            bool bTop = topSet.count(b) > 0;
            if (aBot && bTop) bottomToTop[a] = b;
            else if (bBot && aTop)
                bottomToTop[b] = a;
        }
    }

    std::vector<localIdx> nodes(6);
    for (int i = 0; i < 3; ++i)
    {
        auto bi = static_cast<std::size_t>(i);
        nodes[bi] = bottom[bi];
        auto it = bottomToTop.find(bottom[bi]);
        if (it != bottomToTop.end())
        {
            nodes[bi + 3] = it->second;
        }
    }
    return nodes;
}


CellConnectivity extractCellConnectivity(vtkUnstructuredGrid* grid)
{
    CellConnectivity conn;

    auto* iter = grid->NewCellIterator();
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
    {
        int cellType = iter->GetCellType();

        // Only process 3D cells
        if (cellType != VTK_TETRA && cellType != VTK_HEXAHEDRON && cellType != VTK_WEDGE
            && cellType != VTK_PYRAMID)
        {
            continue;
        }

        vtkIdList* ptIds = iter->GetPointIds();
        std::vector<localIdx> nodes;
        nodes.reserve(static_cast<std::size_t>(ptIds->GetNumberOfIds()));
        for (vtkIdType i = 0; i < ptIds->GetNumberOfIds(); ++i)
        {
            nodes.push_back(static_cast<localIdx>(ptIds->GetId(i)));
        }

        conn.cellToNodes.push_back(std::move(nodes));
        conn.cellTypes.push_back(cellType);
    }
    iter->Delete();

    conn.nCells = static_cast<localIdx>(conn.cellToNodes.size());
    return conn;
}


} // namespace NeoN::io
