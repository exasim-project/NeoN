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

} // namespace NeoN::io
