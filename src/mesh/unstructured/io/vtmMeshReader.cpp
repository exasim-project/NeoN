// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/vtmMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <vtkIdList.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLMultiBlockDataReader.h>

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


namespace NeoN::io
{

namespace
{

/** Convert a Vector<localIdx> to a Vector<label> (may differ in signedness/width). */
labelVector toLabel(const Executor& exec, const Vector<localIdx>& v)
{
    auto h = v.copyToHost();
    std::vector<label> buf(static_cast<std::size_t>(h.size()));
    for (std::size_t i = 0; i < buf.size(); ++i)
    {
        buf[i] = static_cast<label>(h.view()[static_cast<localIdx>(i)]);
    }
    return labelVector(exec, buf);
}

/**
 * Build a map from sorted face-node key → patch index, for every face in
 * every sub-block of @p boundary.  Used during boundary-face matching.
 */
std::map<std::vector<localIdx>, localIdx> buildFaceKeyToPatch(vtkMultiBlockDataSet* boundary)
{
    std::map<std::vector<localIdx>, localIdx> keyMap;
    if (!boundary)
    {
        return keyMap;
    }
    for (unsigned int b = 0; b < boundary->GetNumberOfBlocks(); ++b)
    {
        auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(b));
        if (!patch)
        {
            continue;
        }
        for (vtkIdType f = 0; f < patch->GetNumberOfCells(); ++f)
        {
            vtkNew<vtkIdList> ids;
            patch->GetCellPoints(f, ids);
            std::vector<localIdx> key(static_cast<std::size_t>(ids->GetNumberOfIds()));
            for (vtkIdType n = 0; n < ids->GetNumberOfIds(); ++n)
            {
                key[static_cast<std::size_t>(n)] = static_cast<localIdx>(ids->GetId(n));
            }
            std::sort(key.begin(), key.end());
            keyMap[key] = static_cast<localIdx>(b);
        }
    }
    return keyMap;
}

} // namespace


UnstructuredMesh readVtm(const std::string& filePath, const Executor& exec)
{
    vtkNew<vtkXMLMultiBlockDataReader> reader;
    reader->SetFileName(filePath.c_str());
    reader->Update();

    auto* mb = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
    if (!mb)
    {
        throw std::runtime_error("Failed to read VTM file: " + filePath);
    }

    // --- Block 0: internalMesh ---
    auto* grid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
    if (!grid)
    {
        throw std::runtime_error("Block 0 is not a vtkUnstructuredGrid in: " + filePath);
    }

    // --- Block 1: boundary patches ---
    auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
    const std::vector<std::string> patchNames = multiBlockPatchNames(boundary);
    const auto faceKeyToPatch = buildFaceKeyToPatch(boundary);
    const localIdx nPatches = boundary ? static_cast<localIdx>(boundary->GetNumberOfBlocks()) : 0;

    // --- Extract points ---
    const localIdx nPoints = static_cast<localIdx>(grid->GetNumberOfPoints());
    std::vector<Vec3> hostPoints(static_cast<std::size_t>(nPoints));
    for (localIdx i = 0; i < nPoints; ++i)
    {
        double p[3];
        grid->GetPoint(static_cast<vtkIdType>(i), p);
        hostPoints[static_cast<std::size_t>(i)] =
            Vec3 {static_cast<scalar>(p[0]), static_cast<scalar>(p[1]), static_cast<scalar>(p[2])};
    }

    // --- Reconstruct face topology and geometry ---
    SerialExecutor serial;
    auto conn = extractCellConnectivity(grid, serial);
    const localIdx nCells = conn.nCells;
    auto topo = buildFaceTopology(serial, conn);

    const localIdx nInternalFaces = topo.nInternalFaces;
    const localIdx nBoundaryFaces = topo.nBoundaryFaces;

    auto fnCopyForGeom = topo.faceNodes;
    auto geom = computeGeometry(
        serial,
        Vector<Vec3>(serial, hostPoints),
        topo.faceOwner,
        topo.faceNeighbour,
        fnCopyForGeom,
        nInternalFaces,
        nCells
    );

    // --- Assign reconstructed boundary faces to VTM patches ---
    // Match by sorted face-node set, then build a permutation that groups
    // boundary faces by ascending patch index for the patchOffsets layout.
    std::vector<localIdx> patchOffsets;
    std::vector<localIdx> boundaryFacePerm(static_cast<std::size_t>(nBoundaryFaces));

    if (nPatches == 0 || faceKeyToPatch.empty())
    {
        // No patch metadata: all boundary faces form a single patch.
        patchOffsets = {0, nBoundaryFaces};
        std::iota(boundaryFacePerm.begin(), boundaryFacePerm.end(), localIdx {0});
    }
    else
    {
        auto fnHost = topo.faceNodes.copyToHost();
        auto fnView = fnHost.view();

        std::vector<localIdx> faceToPatch(
            static_cast<std::size_t>(nBoundaryFaces), static_cast<localIdx>(-1)
        );
        for (localIdx i = 0; i < nBoundaryFaces; ++i)
        {
            const localIdx fi = nInternalFaces + i;
            auto [start, end] = fnView.bounds(fi);
            std::vector<localIdx> key;
            key.reserve(static_cast<std::size_t>(end - start));
            for (auto n = start; n < end; ++n)
            {
                key.push_back(fnView.values[n]);
            }
            std::sort(key.begin(), key.end());
            auto it = faceKeyToPatch.find(key);
            if (it != faceKeyToPatch.end())
            {
                faceToPatch[static_cast<std::size_t>(i)] = it->second;
            }
        }

        // Count faces per patch and build offset array.
        std::vector<localIdx> patchCount(static_cast<std::size_t>(nPatches), 0);
        for (localIdx i = 0; i < nBoundaryFaces; ++i)
        {
            const localIdx p = faceToPatch[static_cast<std::size_t>(i)];
            if (p < nPatches)
            {
                patchCount[static_cast<std::size_t>(p)]++;
            }
        }
        patchOffsets.resize(static_cast<std::size_t>(nPatches + 1));
        patchOffsets[0] = 0;
        for (localIdx p = 0; p < nPatches; ++p)
        {
            patchOffsets[static_cast<std::size_t>(p + 1)] =
                patchOffsets[static_cast<std::size_t>(p)] + patchCount[static_cast<std::size_t>(p)];
        }

        // Build permutation: group boundary faces by patch.
        std::vector<localIdx> insertPos(patchOffsets.begin(), patchOffsets.end() - 1);
        for (localIdx i = 0; i < nBoundaryFaces; ++i)
        {
            const localIdx p = faceToPatch[static_cast<std::size_t>(i)];
            if (p < nPatches)
            {
                boundaryFacePerm[static_cast<std::size_t>(insertPos[static_cast<std::size_t>(p)]++
                )] = i;
            }
        }
    }

    // --- Apply permutation to face owner and face-node arrays ---
    // Internal faces keep their order; boundary faces are reordered by patch.
    auto owHost = topo.faceOwner.copyToHost();
    auto fnHost = topo.faceNodes.copyToHost();
    auto fnView = fnHost.view();

    const std::size_t nFacesTotal = static_cast<std::size_t>(nInternalFaces + nBoundaryFaces);
    std::vector<localIdx> newOwner(nFacesTotal);
    for (localIdx i = 0; i < nInternalFaces; ++i)
    {
        newOwner[static_cast<std::size_t>(i)] = owHost.view()[i];
    }
    for (localIdx i = 0; i < nBoundaryFaces; ++i)
    {
        const localIdx oldIdx = boundaryFacePerm[static_cast<std::size_t>(i)];
        newOwner[static_cast<std::size_t>(nInternalFaces + i)] =
            owHost.view()[nInternalFaces + oldIdx];
    }

    std::vector<localIdx> newFnValues;
    std::vector<localIdx> newFnSizes;
    newFnValues.reserve(fnView.values.size());
    newFnSizes.reserve(nFacesTotal);

    for (localIdx fi = 0; fi < nInternalFaces; ++fi)
    {
        auto [s, e] = fnView.bounds(fi);
        newFnSizes.push_back(e - s);
        for (auto n = s; n < e; ++n)
        {
            newFnValues.push_back(fnView.values[n]);
        }
    }
    for (localIdx i = 0; i < nBoundaryFaces; ++i)
    {
        const localIdx fi = nInternalFaces + boundaryFacePerm[static_cast<std::size_t>(i)];
        auto [s, e] = fnView.bounds(fi);
        newFnSizes.push_back(e - s);
        for (auto n = s; n < e; ++n)
        {
            newFnValues.push_back(fnView.values[n]);
        }
    }

    // Build offset array for permuted faceNodes.
    std::vector<localIdx> fnOffsets;
    fnOffsets.reserve(newFnSizes.size() + 1);
    fnOffsets.push_back(0);
    for (localIdx sz : newFnSizes)
    {
        fnOffsets.push_back(fnOffsets.back() + sz);
    }

    // Permuted face arrays on the target executor.
    Vector<localIdx> permFaceOwnerLocal(exec, newOwner);
    auto permFaceNeighbourLocal = topo.faceNeighbour.copyToExecutor(exec);
    SegmentedVector<localIdx, localIdx> permFaceNodes(
        Vector<localIdx>(serial, newFnValues).copyToExecutor(exec),
        Vector<localIdx>(serial, fnOffsets).copyToExecutor(exec)
    );

    // Recompute geometry with permuted order so face arrays are consistent.
    auto permFaceNodesCopy = permFaceNodes;
    auto geomPerm = computeGeometry(
        exec,
        Vector<Vec3>(exec, hostPoints),
        permFaceOwnerLocal,
        permFaceNeighbourLocal,
        permFaceNodesCopy,
        nInternalFaces,
        nCells
    );

    // --- Build BoundaryMesh and UnstructuredMesh ---
    // Field-name mapping (feat/mesh_io geometry struct → develop constructor):
    //   geomPerm.faceAreas    = area vectors  → faceNormals param
    //   geomPerm.magFaceAreas = area mags     → faceAreas param
    //   geomPerm.faceCenters  = face centers  → faceCenters param
    //   geomPerm.cellCenters  = cell centers  → cellCenters param

    auto boundaryMesh = buildBoundaryMesh(
        exec,
        permFaceOwnerLocal,
        geomPerm.faceCenters,
        geomPerm.cellCenters,
        geomPerm.faceAreas,
        geomPerm.magFaceAreas,
        nInternalFaces,
        nBoundaryFaces,
        patchOffsets
    );

    // UnstructuredMesh on develop uses labelVector (Vector<label>) which may
    // differ from Vector<localIdx>; convert explicitly.
    const labelVector faceOwnersLabel = toLabel(exec, permFaceOwnerLocal);
    const labelVector faceNeighborsLabel = toLabel(exec, permFaceNeighbourLocal);

    UnstructuredMesh mesh(
        exec,
        vectorVector(exec, hostPoints),
        geomPerm.cellVolumes,
        geomPerm.cellCenters,
        geomPerm.faceAreas,
        geomPerm.faceCenters,
        geomPerm.magFaceAreas,
        faceOwnersLabel,
        faceNeighborsLabel,
        std::move(boundaryMesh)
    );

    // --- Store stencilDB metadata for round-trip write-back ---
    mesh.stencilDB().insert(
        std::string(stencilFaceNodes),
        std::make_shared<SegmentedVector<localIdx, localIdx>>(std::move(permFaceNodes))
    );
    if (!patchNames.empty())
    {
        mesh.stencilDB().insert(
            std::string(stencilPatchNames), std::make_shared<std::vector<std::string>>(patchNames)
        );
    }

    return mesh;
}

} // namespace NeoN::io
