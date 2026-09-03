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

namespace NeoN::io
{

/** Build a cell-to-face mapping as a SegmentedVector. */
SegmentedVector<localIdx, localIdx> buildCellToFaceMapping(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    localIdx nInternalFaces,
    localIdx nCells
);

/** Compute cell centres as average of face centres per cell. */
Vector<Vec3> computeCellCentres(
    const Executor& exec,
    const Vector<Vec3>& faceCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
);

/** Compute cell volumes via tetrahedral decomposition. */
Vector<scalar> computeCellVolumes(
    const Executor& exec,
    const Vector<Vec3>& points,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    const Vector<Vec3>& faceCentres,
    const Vector<Vec3>& cellCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
);

} // namespace NeoN::io
