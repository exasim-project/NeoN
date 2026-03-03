// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/meshConverter.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataAssembly.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>

#include <vtkIntArray.h>

#include <set>
#include <vector>


namespace NeoN::io
{

namespace
{

/// Data extracted from mesh, shared by buildMultiBlockMesh and buildPartitionedMesh.
struct MeshBuildData
{
    vtkSmartPointer<vtkPoints> vtkPts;
    vtkSmartPointer<vtkUnstructuredGrid> volumeGrid;
    std::vector<std::string> patchNames;
    std::vector<vtkSmartPointer<vtkPolyData>> patches;
};

MeshBuildData buildMeshData(const UnstructuredMesh& mesh)
{
    localIdx nCells = mesh.nCells();
    localIdx nInternalFaces = mesh.nInternalFaces();
    localIdx nFaces = mesh.nFaces();

    // Copy data to host
    auto hostPoints = mesh.points().copyToHost();
    auto hostFaceOwner = mesh.faceOwner().copyToHost();
    auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
    localIdx nPoints = hostPoints.size();

    // Retrieve face node connectivity from stencilDB
    if (!mesh.stencilDB().contains("io::faceNodes"))
    {
        throw std::runtime_error("buildMeshData: face node connectivity not available in stencilDB."
        );
    }
    auto& faceNodes =
        *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>("io::faceNodes");

    // Convert faceOwner/faceNeighbour to std::vector
    std::vector<label> faceOwnerVec(static_cast<std::size_t>(nFaces));
    for (localIdx i = 0; i < nFaces; ++i)
    {
        faceOwnerVec[static_cast<std::size_t>(i)] = hostFaceOwner.view()[i];
    }
    std::vector<label> faceNeighbourVec(static_cast<std::size_t>(nInternalFaces));
    for (localIdx i = 0; i < nInternalFaces; ++i)
    {
        faceNeighbourVec[static_cast<std::size_t>(i)] = hostFaceNeighbour.view()[i];
    }

    // Rebuild cell info
    auto cells =
        rebuildCellInfo(faceOwnerVec, faceNeighbourVec, faceNodes, nCells, nInternalFaces, nFaces);

    // Build shared VTK points
    vtkNew<vtkPoints> vtkPts;
    vtkPts->SetNumberOfPoints(nPoints);
    for (localIdx i = 0; i < nPoints; ++i)
    {
        const auto& p = hostPoints.view()[i];
        vtkPts->SetPoint(
            i, static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])
        );
    }

    // Build volume grid
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

    // Build boundary patches
    auto const& offset = mesh.boundaryMesh().offset();
    localIdx nBoundaries = mesh.nBoundaries();

    std::vector<std::string> patchNames;
    if (mesh.stencilDB().contains("io::patchNames"))
    {
        auto& names =
            mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
        patchNames = *names;
    }

    std::vector<vtkSmartPointer<vtkPolyData>> patches;
    std::vector<std::string> resolvedNames;
    for (localIdx b = 0; b < nBoundaries; ++b)
    {
        localIdx patchStart = offset[static_cast<std::size_t>(b)];
        localIdx patchEnd = offset[static_cast<std::size_t>(b + 1)];

        vtkNew<vtkPolyData> polyData;
        polyData->SetPoints(vtkPts);

        vtkNew<vtkCellArray> polys;
        for (localIdx i = patchStart; i < patchEnd; ++i)
        {
            auto fi = static_cast<std::size_t>(nInternalFaces + i);
            const auto& fn = faceNodes[fi];

            std::vector<vtkIdType> pts;
            pts.reserve(fn.size());
            for (localIdx n : fn)
            {
                pts.push_back(static_cast<vtkIdType>(n));
            }
            polys->InsertNextCell(static_cast<int>(pts.size()), pts.data());
        }
        polyData->SetPolys(polys);
        patches.push_back(polyData);

        std::string name = (static_cast<std::size_t>(b) < patchNames.size())
                             ? patchNames[static_cast<std::size_t>(b)]
                             : "patch_" + std::to_string(b);
        resolvedNames.push_back(name);
    }

    MeshBuildData data;
    data.vtkPts = vtkPts;
    data.volumeGrid = grid;
    data.patchNames = resolvedNames;
    data.patches = patches;
    return data;
}

} // anonymous namespace


vtkSmartPointer<vtkMultiBlockDataSet>
buildMultiBlockMesh(const UnstructuredMesh& mesh, bool includeGhosts)
{
    auto data = buildMeshData(mesh);

    vtkIdType nRealCells = data.volumeGrid->GetNumberOfCells();
    vtkIdType nGhostCells = 0;

    // Add ghost cells to volume grid if requested and available
    if (includeGhosts && mesh.stencilDB().contains("partition::ghostCellFaceNodes"))
    {
        auto& ghostFaceNodes =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<std::vector<localIdx>>>>>(
                "partition::ghostCellFaceNodes"
            );
        auto& ghostPts =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<Vec3>>>("partition::ghostPoints");

        // Add ghost-only points to VTK points
        for (const auto& p : ghostPts)
        {
            data.vtkPts->InsertNextPoint(
                static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])
            );
        }

        // Add ghost cells using their face-node connectivity
        for (std::size_t gc = 0; gc < ghostFaceNodes.size(); ++gc)
        {
            const auto& cellFaces = ghostFaceNodes[gc];
            if (cellFaces.empty()) continue;

            // Collect unique nodes for this ghost cell
            std::set<localIdx> nodeSet;
            for (const auto& face : cellFaces)
                for (localIdx n : face)
                    nodeSet.insert(n);

            // Build CellInfo for node ordering
            CellInfo info;
            info.nodeIds.assign(nodeSet.begin(), nodeSet.end());
            info.cellFaceNodes = cellFaces;

            localIdx nNodes = static_cast<localIdx>(info.nodeIds.size());
            localIdx nFacesPerCell = static_cast<localIdx>(cellFaces.size());

            // Determine VTK cell type from face/node count
            if (nFacesPerCell == 4 && nNodes == 4) info.cellType = 9; // VTK_QUAD
            else if (nFacesPerCell == 4 && nNodes == 4)
                info.cellType = 10; // VTK_TETRA
            else if (nFacesPerCell == 6 && nNodes == 8)
                info.cellType = 12; // VTK_HEXAHEDRON
            else if (nFacesPerCell == 5 && nNodes == 5)
                info.cellType = 14; // VTK_PYRAMID
            else if (nFacesPerCell == 5 && nNodes == 6)
                info.cellType = 13; // VTK_WEDGE
            else
                info.cellType = 12; // fallback to hex

            std::vector<localIdx> ordered;
            switch (info.cellType)
            {
            case 9:
                ordered = orderQuadNodes(info);
                break;
            case 10:
                ordered = orderTetNodes(info);
                break;
            case 12:
                ordered = orderHexNodes(info);
                break;
            case 14:
                ordered = orderPyramidNodes(info);
                break;
            case 13:
                ordered = orderWedgeNodes(info);
                break;
            default:
                ordered = info.nodeIds;
                break;
            }

            std::vector<vtkIdType> pts;
            pts.reserve(ordered.size());
            for (localIdx n : ordered)
                pts.push_back(static_cast<vtkIdType>(n));

            data.volumeGrid->InsertNextCell(
                info.cellType, static_cast<int>(pts.size()), pts.data()
            );
            ++nGhostCells;
        }

        // Add "ghostCells" cell data: 0 for real cells, 1 for ghost cells
        vtkNew<vtkIntArray> ghostArray;
        ghostArray->SetName("ghostCells");
        ghostArray->SetNumberOfTuples(nRealCells + nGhostCells);
        for (vtkIdType i = 0; i < nRealCells; ++i)
            ghostArray->SetValue(i, 0);
        for (vtkIdType i = nRealCells; i < nRealCells + nGhostCells; ++i)
            ghostArray->SetValue(i, 1);
        data.volumeGrid->GetCellData()->AddArray(ghostArray);
    }

    vtkSmartPointer<vtkMultiBlockDataSet> mb = vtkSmartPointer<vtkMultiBlockDataSet>::New();

    // Block 0: internalMesh
    mb->SetBlock(0, data.volumeGrid);
    mb->GetMetaData(0u)->Set(vtkCompositeDataSet::NAME(), "internalMesh");

    // Block 1: boundary (nested multiblock containing all patches)
    vtkSmartPointer<vtkMultiBlockDataSet> boundary = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    for (std::size_t b = 0; b < data.patches.size(); ++b)
    {
        auto patchIdx = static_cast<unsigned int>(b);
        boundary->SetBlock(patchIdx, data.patches[b]);
        boundary->GetMetaData(patchIdx)->Set(
            vtkCompositeDataSet::NAME(), data.patchNames[b].c_str()
        );
    }

    mb->SetBlock(1, boundary);
    mb->GetMetaData(1u)->Set(vtkCompositeDataSet::NAME(), "boundary");

    return mb;
}


std::vector<std::string> multiBlockPatchNames(vtkMultiBlockDataSet* boundary)
{
    std::vector<std::string> names;
    if (!boundary)
    {
        return names;
    }
    for (unsigned int i = 0; i < boundary->GetNumberOfBlocks(); ++i)
    {
        if (boundary->HasMetaData(i) && boundary->GetMetaData(i)->Has(vtkCompositeDataSet::NAME()))
        {
            names.emplace_back(boundary->GetMetaData(i)->Get(vtkCompositeDataSet::NAME()));
        }
        else
        {
            names.emplace_back("patch_" + std::to_string(i));
        }
    }
    return names;
}


vtkSmartPointer<vtkPartitionedDataSetCollection> buildPartitionedMesh(const UnstructuredMesh& mesh)
{
    auto data = buildMeshData(mesh);

    vtkNew<vtkPartitionedDataSetCollection> pdc;

    // Dataset 0: volume grid
    vtkNew<vtkPartitionedDataSet> volPds;
    volPds->SetPartition(0, data.volumeGrid);
    pdc->SetPartitionedDataSet(0, volPds);

    // Datasets 1..N: boundary patches
    for (std::size_t b = 0; b < data.patches.size(); ++b)
    {
        auto dsIdx = static_cast<unsigned int>(b + 1);
        vtkNew<vtkPartitionedDataSet> patchPds;
        patchPds->SetPartition(0, data.patches[b]);
        pdc->SetPartitionedDataSet(dsIdx, patchPds);
    }

    // Build assembly hierarchy
    vtkNew<vtkDataAssembly> assembly;
    assembly->Initialize();
    int root = assembly->GetRootNode();
    int meshNode = assembly->AddNode("internalMesh", root);
    assembly->AddDataSetIndex(meshNode, 0);
    int bndNode = assembly->AddNode("boundary", root);
    for (std::size_t b = 0; b < data.patches.size(); ++b)
    {
        int patchNode = assembly->AddNode(data.patchNames[b].c_str(), bndNode);
        assembly->AddDataSetIndex(patchNode, static_cast<unsigned int>(b + 1));
    }
    pdc->SetDataAssembly(assembly);

    return pdc.Get();
}


} // namespace NeoN::io
