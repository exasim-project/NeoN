// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/vtuMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"

#include <vtkCellArray.h>
#include <vtkCellType.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace NeoN::io
{

void writeVtu(const UnstructuredMesh& mesh, const std::string& filePath)
{
    localIdx nCells = mesh.nCells();
    localIdx nInternalFaces = mesh.nInternalFaces();
    localIdx nFaces = mesh.nFaces();

    // Copy points to host
    auto hostPoints = mesh.points().copyToHost();
    localIdx nPoints = hostPoints.size();

    // Retrieve face node connectivity from stencilDB
    if (!mesh.stencilDB().contains(std::string(stencilFaceNodes)))
    {
        throw std::runtime_error("writeVtu: face node connectivity not available in stencilDB. "
                                 "Only meshes created by readCgns can be written.");
    }
    auto& faceNodes = *mesh.stencilDB().get<std::shared_ptr<SegmentedVector<localIdx, localIdx>>>(
        std::string(stencilFaceNodes)
    );

    // Rebuild cell info (rebuildCellInfo copies inputs to host internally)
    auto cells = rebuildCellInfo(
        mesh.faceOwner(), mesh.faceNeighbour(), faceNodes, nCells, nInternalFaces, nFaces
    );

    // Build VTK points
    vtkNew<vtkPoints> vtkPts;
    vtkPts->SetNumberOfPoints(nPoints);
    for (localIdx i = 0; i < nPoints; ++i)
    {
        const auto& p = hostPoints.view()[i];
        vtkPts->SetPoint(
            i, static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])
        );
    }

    // Build VTK cells using shared node ordering
    vtkNew<vtkUnstructuredGrid> grid;
    grid->SetPoints(vtkPts);
    grid->Allocate(nCells);

    for (std::size_t c = 0; c < cells.size(); ++c)
    {
        auto& cell = cells[c];
        std::vector<localIdx> ordered;

        switch (cell.cellType)
        {
        case 9: // VTK_QUAD
            ordered = orderQuadNodes(cell);
            break;
        case 10: // VTK_TETRA
            ordered = orderTetNodes(cell);
            break;
        case 12: // VTK_HEXAHEDRON
            ordered = orderHexNodes(cell);
            break;
        case 14: // VTK_PYRAMID
            ordered = orderPyramidNodes(cell);
            break;
        case 13: // VTK_WEDGE
            ordered = orderWedgeNodes(cell);
            break;
        default:
            ordered = cell.nodeIds;
            break;
        }

        std::vector<vtkIdType> pts;
        pts.reserve(ordered.size());
        for (localIdx n : ordered)
        {
            pts.push_back(static_cast<vtkIdType>(n));
        }
        grid->InsertNextCell(cell.cellType, static_cast<int>(pts.size()), pts.data());
    }

    // Write VTU
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(grid);
    writer->Write();
}

} // namespace NeoN::io
