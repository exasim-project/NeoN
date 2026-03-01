// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/connectivity/nodeOrdering.hpp"

#include <map>
#include <set>
#include <vector>


namespace NeoN::io
{

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


} // namespace NeoN::io
