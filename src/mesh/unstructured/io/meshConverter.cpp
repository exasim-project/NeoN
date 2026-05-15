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

    // Copy points data to host
    auto hostPoints = mesh.points().copyToHost();
    localIdx nPoints = hostPoints.size();

    // Retrieve face node connectivity from stencilDB
    if (!mesh.stencilDB().contains(std::string(stencilFaceNodes)))
    {
        throw std::runtime_error("buildMeshData: face node connectivity not available in stencilDB."
        );
    }
    auto& faceNodes =
        *mesh.stencilDB().get<std::shared_ptr<SegmentedVector<localIdx, localIdx>>>(std::string(stencilFaceNodes)
        );

    // Rebuild cell info (rebuildCellInfo copies inputs to host internally)
    auto cells = rebuildCellInfo(
        mesh.faceOwner(), mesh.faceNeighbour(), faceNodes, nCells, nInternalFaces, nFaces
    );

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
    if (mesh.stencilDB().contains(std::string(stencilPatchNames)))
    {
        auto& names =
            mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>(std::string(stencilPatchNames));
        patchNames = *names;
    }

    // Access faceNodes via host view (always stored on SerialExecutor)
    auto hostFaceNodes = faceNodes.copyToHost();
    auto fnView = hostFaceNodes.view();

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
            localIdx fi = nInternalFaces + i;
            auto [start, end] = fnView.bounds(fi);

            std::vector<vtkIdType> pts;
            pts.reserve(static_cast<std::size_t>(end - start));
            for (auto n = start; n < end; ++n)
            {
                pts.push_back(static_cast<vtkIdType>(fnView.values[n]));
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


BoundaryMesh buildBoundaryMesh(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<Vec3>& faceCentres,
    const Vector<Vec3>& cellCentres,
    const Vector<Vec3>& faceAreas,
    const Vector<scalar>& magFaceAreas,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    const std::vector<localIdx>& patchOffsets
)
{
    auto owView = faceOwner.view();
    auto hFaceCentres = faceCentres.view();
    auto hCellCentres = cellCentres.view();
    auto hFaceAreas = faceAreas.view();
    auto hMagFaceAreas = magFaceAreas.view();

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

        localIdx ownerCell = owView[fi];

        bndFaceCells[bi] = static_cast<label>(ownerCell);
        bndCf[bi] = hFaceCentres[fi];
        bndCn[bi] = hCellCentres[ownerCell];
        bndSf[bi] = hFaceAreas[fi];
        bndMagSf[bi] = hMagFaceAreas[fi];

        if (bndMagSf[bi] > 1e-30)
        {
            bndNf[bi] = bndSf[bi] * (1.0 / bndMagSf[bi]);
        }
        else
        {
            bndNf[bi] = Vec3 {0.0, 0.0, 0.0};
        }

        bndDelta[bi] = bndCf[bi] - bndCn[bi];

        scalar magDelta = mag(bndDelta[bi]);
        bndDeltaCoeffs[bi] = (magDelta > 1e-30) ? 1.0 / magDelta : 0.0;

        bndWeights[bi] = 1.0;
    }

    return BoundaryMesh(
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
}


vtkSmartPointer<vtkMultiBlockDataSet> buildMultiBlockMesh(const UnstructuredMesh& mesh)
{
    auto data = buildMeshData(mesh);

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
