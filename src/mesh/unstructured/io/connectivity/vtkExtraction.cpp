// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/connectivity/vtkExtraction.hpp"
#include "detail.hpp"

#include <vtkCellIterator.h>
#include <vtkCellType.h>
#include <vtkIdList.h>
#include <vtkUnstructuredGrid.h>

#include <vector>


namespace NeoN::io
{

CellConnectivity extractCellConnectivity(vtkUnstructuredGrid* grid, const Executor& exec)
{
    std::vector<std::vector<localIdx>> hostCellToNodes;
    std::vector<int32_t> hostCellTypes;

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

        hostCellToNodes.push_back(std::move(nodes));
        hostCellTypes.push_back(static_cast<int32_t>(cellType));
    }
    iter->Delete();

    localIdx nCells = static_cast<localIdx>(hostCellToNodes.size());

    std::vector<localIdx> cellNodeValues;
    std::vector<localIdx> cellNodeSizes;
    for (const auto& nodes : hostCellToNodes)
    {
        cellNodeSizes.push_back(static_cast<localIdx>(nodes.size()));
        cellNodeValues.insert(cellNodeValues.end(), nodes.begin(), nodes.end());
    }

    return CellConnectivity {
        detail::makeSegmentedVector(cellNodeValues, cellNodeSizes, exec),
        Vector<int32_t>(SerialExecutor {}, hostCellTypes).copyToExecutor(exec),
        nCells
    };
}


} // namespace NeoN::io
