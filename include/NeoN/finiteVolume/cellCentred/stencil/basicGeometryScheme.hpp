// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/** @class BasicGeometryScheme
 *  @brief Default GeometryScheme kernel: computes weights, nonOrthDeltaCoeffs and the
 *  non-orthogonal correction vectors directly from the mesh geometry.
 *
 *  Scheme definitions baked into this kernel:
 *   - nonOrthDeltaCoeffs is the over-relaxed inverse face-normal distance 1/(n.d), floored by
 *     nonOrthDeltaClamp; the snGrad schemes consume it as the consistent normal-gradient
 *     coefficient on non-orthogonal meshes (it equals the Euclidean 1/|d| on orthogonal meshes).
 *   - boundary weights are 1 (one-sided physical patches); processor weights are dN/(dO+dN).
 *   - processor nonOrthDeltaCoeffs are 1/(dO+dN) from the exchanged owner face-normal distances.
 *   - non-orthogonal correction vectors are zero on physical (one-sided) boundary faces, but
 *     non-zero on non-orthogonal processor faces: a processor face has a real neighbour cell, so
 *     the corrected/limited snGrad applies the full correction there (it vanishes on orthogonal
 *     processor faces). See FaceNormalGradient docs.
 *
 *  Marked final to enable devirtualisation of the kernel calls in GeometryScheme::update().
 */
class BasicGeometryScheme final : public GeometrySchemeFactory
{

public:

    BasicGeometryScheme(const UnstructuredMesh& mesh);

    void updateWeights(const Executor& exec, SurfaceField<scalar>& weights) final;

    void
    updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs) final;

    void updateNonOrthCorrectionVec3s(
        const Executor& exec,
        SurfaceField<Vec3>& nonOrthCorrectionVec3s,
        const SurfaceField<scalar>& nonOrthDeltaCoeffs
    ) final;


private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN
