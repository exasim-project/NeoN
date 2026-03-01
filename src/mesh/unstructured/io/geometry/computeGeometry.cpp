// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/geometry/computeGeometry.hpp"
#include "NeoN/mesh/unstructured/io/geometry/cellGeometry.hpp"
#include "NeoN/mesh/unstructured/io/geometry/faceGeometry.hpp"


namespace NeoN::io
{

MeshGeometry computeGeometry(
    const Executor& exec,
    const Vector<Vec3>& points,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    localIdx nInternalFaces,
    localIdx nCells
)
{
    auto faceCentres = computeFaceCentres(exec, points, faceNodes);
    auto faceAreas = computeFaceAreas(exec, points, faceNodes, faceCentres);
    auto magFaceAreasVec = computeMagFaceAreas(exec, faceAreas);

    auto cellFaces = buildCellToFaceMapping(exec, faceOwner, faceNeighbour, nInternalFaces, nCells);
    auto cellCentres = computeCellCentres(exec, faceCentres, cellFaces, nCells);
    auto cellVolumes =
        computeCellVolumes(exec, points, faceNodes, faceCentres, cellCentres, cellFaces, nCells);

    return MeshGeometry {
        std::move(cellVolumes),
        std::move(cellCentres),
        std::move(faceAreas),
        std::move(faceCentres),
        std::move(magFaceAreasVec)
    };
}


MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells)
{
    SerialExecutor exec;

    // Convert points to NeoN Vector
    Vector<Vec3> devicePoints(exec, points);

    // faceNodes needs a non-const copy since computeGeometry takes SegmentedVector&.
    auto faceNodesCopy = topo.faceNodes;

    return computeGeometry(
        exec,
        devicePoints,
        topo.faceOwner,
        topo.faceNeighbour,
        faceNodesCopy,
        topo.nInternalFaces,
        nCells
    );
}


} // namespace NeoN::io
