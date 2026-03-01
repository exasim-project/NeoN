// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/vtkHdfMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <vtkHDFReader.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace NeoN::io
{

UnstructuredMesh readVtkHdf(const std::string& filePath, const Executor& exec)
{
    vtkNew<vtkHDFReader> reader;
    reader->SetFileName(filePath.c_str());
    reader->Update();

    auto* output = reader->GetOutput();
    if (!output)
    {
        throw std::runtime_error("Failed to read VTK HDF file: " + filePath);
    }

    vtkUnstructuredGrid* grid = vtkUnstructuredGrid::SafeDownCast(output);
    if (!grid)
    {
        auto* pdc = vtkPartitionedDataSetCollection::SafeDownCast(output);
        if (pdc && pdc->GetNumberOfPartitionedDataSets() > 0)
        {
            auto* pds0 = pdc->GetPartitionedDataSet(0);
            if (pds0 && pds0->GetNumberOfPartitions() > 0)
            {
                grid = vtkUnstructuredGrid::SafeDownCast(pds0->GetPartition(0));
            }
        }
    }
    if (!grid)
    {
        auto* mb = vtkMultiBlockDataSet::SafeDownCast(output);
        if (mb && mb->GetNumberOfBlocks() > 0)
        {
            grid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        }
    }
    if (!grid)
    {
        auto* pds = vtkPartitionedDataSet::SafeDownCast(output);
        if (pds && pds->GetNumberOfPartitions() > 0)
        {
            grid = vtkUnstructuredGrid::SafeDownCast(pds->GetPartition(0));
        }
    }

    if (!grid)
    {
        throw std::runtime_error("No vtkUnstructuredGrid found in VTK HDF file: " + filePath);
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

    // Build face topology and geometry via meshConverter
    SerialExecutor serial;
    auto conn = extractCellConnectivity(grid, serial);
    localIdx nCells = conn.nCells;
    auto topo = buildFaceTopology(serial, conn);

    localIdx nInternalFaces = topo.nInternalFaces;
    localIdx nBoundaryFaces = topo.nBoundaryFaces;
    localIdx nFaces = nInternalFaces + nBoundaryFaces;

    // No BC info in VTK HDF: single patch with all boundary faces
    localIdx nBoundaries = (nBoundaryFaces > 0) ? 1 : 0;
    std::vector<localIdx> patchOffsets = {0, nBoundaryFaces};

    // computeGeometry returns MeshGeometry on SerialExecutor (host) — no copyToHost needed.
    auto geom = computeGeometry(hostPoints, topo, nCells);

    // Build NeoN vectors on the target executor
    vectorVector meshPoints(exec, hostPoints);
    auto cellVolumes = geom.cellVolumes.copyToExecutor(exec);
    auto cellCentres = geom.cellCentres.copyToExecutor(exec);
    auto faceAreasVec = geom.faceAreas.copyToExecutor(exec);
    auto faceCentresVec = geom.faceCentres.copyToExecutor(exec);
    auto magFaceAreasVec = geom.magFaceAreas.copyToExecutor(exec);
    auto faceOwnerVec = topo.faceOwner.copyToExecutor(exec);
    auto faceNeighbourVec = topo.faceNeighbour.copyToExecutor(exec);

    // Build BoundaryMesh from host geometry
    auto boundaryMesh = buildBoundaryMesh(
        exec,
        topo.faceOwner,
        geom.faceCentres,
        geom.cellCentres,
        geom.faceAreas,
        geom.magFaceAreas,
        nInternalFaces,
        nBoundaryFaces,
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

    // Store face node connectivity for writer round-trip
    auto faceNodePtr = std::make_shared<SegmentedVector<localIdx, localIdx>>(topo.faceNodes);
    mesh.stencilDB().insert(std::string(stencilFaceNodes), faceNodePtr);

    return mesh;
}

} // namespace NeoN::io
