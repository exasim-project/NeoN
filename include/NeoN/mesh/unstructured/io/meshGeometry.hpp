// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"

#include <vector>

namespace NeoN::io
{

/// Geometric quantities derived from mesh topology and point coordinates.
struct MeshGeometry
{
    Vector<scalar> cellVolumes;
    Vector<Vec3> cellCentres;
    Vector<Vec3> faceAreas;
    Vector<Vec3> faceCentres;
    Vector<scalar> magFaceAreas;
};


/// Compute face centres as the average of face node positions.
Vector<Vec3> computeFaceCentres(
    const Executor& exec, const Vector<Vec3>& points, SegmentedVector<localIdx, localIdx>& faceNodes
);

/// Compute face area vectors via triangulation from face centre.
Vector<Vec3> computeFaceAreas(
    const Executor& exec,
    const Vector<Vec3>& points,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    const Vector<Vec3>& faceCentres
);

/// Compute magnitude of face area vectors.
Vector<scalar> computeMagFaceAreas(const Executor& exec, const Vector<Vec3>& faceAreas);

/// Build a cell-to-face mapping as a SegmentedVector.
SegmentedVector<localIdx, localIdx> buildCellToFaceMapping(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    localIdx nInternalFaces,
    localIdx nCells
);

/// Compute cell centres as average of face centres per cell.
Vector<Vec3> computeCellCentres(
    const Executor& exec,
    const Vector<Vec3>& faceCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
);

/// Compute cell volumes via tetrahedral decomposition.
Vector<scalar> computeCellVolumes(
    const Executor& exec,
    const Vector<Vec3>& points,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    const Vector<Vec3>& faceCentres,
    const Vector<Vec3>& cellCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
);

/// Compute all geometric quantities from device vectors.
MeshGeometry computeGeometry(
    const Executor& exec,
    const Vector<Vec3>& points,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    localIdx nInternalFaces,
    localIdx nCells
);

/// Legacy overload: compute from std::vector and FaceTopology.
MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells);


} // namespace NeoN::io
