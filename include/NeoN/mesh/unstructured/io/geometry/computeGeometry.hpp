// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/io/connectivity/types.hpp"
#include "NeoN/mesh/unstructured/io/geometry/types.hpp"

#include <vector>

namespace NeoN::io
{

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

/// Overload: compute from std::vector and FaceTopology.
MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells);

} // namespace NeoN::io
