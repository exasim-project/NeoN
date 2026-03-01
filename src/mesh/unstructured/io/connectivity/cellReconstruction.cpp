// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/connectivity/cellReconstruction.hpp"
#include "detail.hpp"

#include <algorithm>
#include <set>
#include <vector>


namespace NeoN::io
{

CellConnectivity rebuildCellConnectivity(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    const SegmentedVector<localIdx, localIdx>& faceNodes,
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

    // Copy to host for sequential algorithm
    auto hostOwner = faceOwner.copyToHost();
    auto hostNeighbour = faceNeighbour.copyToHost();
    auto hostFaceNodes = faceNodes.copyToHost();

    auto owView = hostOwner.view();
    auto neiView = hostNeighbour.view();
    auto fnView = hostFaceNodes.view();

    // Collect faces per cell
    std::vector<std::vector<localIdx>> cellFaces(static_cast<std::size_t>(nCells));
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto oi = static_cast<std::size_t>(owView[f]);
        cellFaces[oi].push_back(f);
        if (f < nInternalFaces)
        {
            auto ni = static_cast<std::size_t>(neiView[f]);
            cellFaces[ni].push_back(f);
        }
    }

    std::vector<std::vector<localIdx>> hostCellToNodes(static_cast<std::size_t>(nCells));
    std::vector<int32_t> hostCellTypes(static_cast<std::size_t>(nCells));

    for (localIdx c = 0; c < nCells; ++c)
    {
        auto ci = static_cast<std::size_t>(c);

        std::set<localIdx> nodeSet;
        std::size_t maxNodesPerFace = 0;
        for (localIdx f : cellFaces[ci])
        {
            auto [start, end] = fnView.bounds(f);
            maxNodesPerFace = std::max(maxNodesPerFace, static_cast<std::size_t>(end - start));
            for (auto n = start; n < end; ++n)
            {
                nodeSet.insert(fnView.values[n]);
            }
        }
        hostCellToNodes[ci].assign(nodeSet.begin(), nodeSet.end());

        localIdx nCellFaces = static_cast<localIdx>(cellFaces[ci].size());
        localIdx nCellNodes = static_cast<localIdx>(nodeSet.size());

        if (maxNodesPerFace <= 2) hostCellTypes[ci] = VTK_QUAD_TYPE;
        else if (nCellFaces == 4 && nCellNodes == 4)
            hostCellTypes[ci] = VTK_TETRA_TYPE;
        else if (nCellFaces == 6 && nCellNodes == 8)
            hostCellTypes[ci] = VTK_HEXAHEDRON_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 6)
            hostCellTypes[ci] = VTK_WEDGE_TYPE;
        else if (nCellFaces == 5 && nCellNodes == 5)
            hostCellTypes[ci] = VTK_PYRAMID_TYPE;
        else if (nCellNodes == 4)
            hostCellTypes[ci] = VTK_TETRA_TYPE;
        else if (nCellNodes == 8)
            hostCellTypes[ci] = VTK_HEXAHEDRON_TYPE;
        else
            hostCellTypes[ci] = VTK_TETRA_TYPE;
    }

    // Pack into NeoN types: flatten cellToNodes into SegmentedVector
    std::vector<localIdx> cellNodeValues;
    std::vector<localIdx> cellNodeSizes;
    for (const auto& nodes : hostCellToNodes)
    {
        cellNodeSizes.push_back(static_cast<localIdx>(nodes.size()));
        cellNodeValues.insert(cellNodeValues.end(), nodes.begin(), nodes.end());
    }

    SerialExecutor serial;

    return CellConnectivity {
        detail::makeSegmentedVector(cellNodeValues, cellNodeSizes, exec),
        Vector<int32_t>(serial, hostCellTypes).copyToExecutor(exec),
        nCells
    };
}


std::vector<CellInfo> rebuildCellInfo(
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    const SegmentedVector<localIdx, localIdx>& faceNodes,
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

    // Copy to host for sequential algorithm
    auto hostOwner = faceOwner.copyToHost();
    auto hostNeighbour = faceNeighbour.copyToHost();
    auto hostFaceNodes = faceNodes.copyToHost();

    auto owView = hostOwner.view();
    auto neiView = hostNeighbour.view();
    auto fnView = hostFaceNodes.view();

    // Collect faces per cell
    std::vector<std::vector<localIdx>> cellFaces(static_cast<std::size_t>(nCells));
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto oi = static_cast<std::size_t>(owView[f]);
        cellFaces[oi].push_back(f);
        if (f < nInternalFaces)
        {
            auto ni = static_cast<std::size_t>(neiView[f]);
            cellFaces[ni].push_back(f);
        }
    }

    std::vector<CellInfo> cells(static_cast<std::size_t>(nCells));

    for (localIdx c = 0; c < nCells; ++c)
    {
        auto ci = static_cast<std::size_t>(c);
        auto& cell = cells[ci];

        std::set<localIdx> nodeSet;
        for (localIdx f : cellFaces[ci])
        {
            auto [start, end] = fnView.bounds(f);
            std::vector<localIdx> faceNodeVec;
            faceNodeVec.reserve(static_cast<std::size_t>(end - start));
            for (auto n = start; n < end; ++n)
            {
                faceNodeVec.push_back(fnView.values[n]);
                nodeSet.insert(fnView.values[n]);
            }
            cell.cellFaceNodes.push_back(std::move(faceNodeVec));
        }
        cell.nodeIds.assign(nodeSet.begin(), nodeSet.end());

        localIdx nCellFaces = static_cast<localIdx>(cell.cellFaceNodes.size());
        localIdx nCellNodes = static_cast<localIdx>(cell.nodeIds.size());

        std::size_t maxNodesPerFace = 0;
        for (const auto& fn : cell.cellFaceNodes)
        {
            maxNodesPerFace = std::max(maxNodesPerFace, fn.size());
        }

        if (maxNodesPerFace <= 2) cell.cellType = VTK_QUAD_TYPE;
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


} // namespace NeoN::io
