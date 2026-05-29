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
 *  @brief Default GeometryScheme kernel: computes weights, deltaCoeffs, nonOrthDeltaCoeffs and
 *  the non-orthogonal correction vectors directly from the mesh geometry.
 *
 *  Assumptions baked into this kernel (review L8):
 *   - deltaCoeffs is the orthogonal inverse cell-to-cell distance 1/|d| (matches OpenFOAM's
 *     uncorrectedSnGrad); nonOrthDeltaCoeffs is the over-relaxed 1/(n.d) floored by
 *     nonOrthDeltaClamp.
 *   - boundary weights are 1 (one-sided physical patches); processor weights are dN/(dO+dN).
 *   - processor-boundary deltaCoeffs/nonOrthDeltaCoeffs use the face-normal-projected
 *     owner+neighbour distance (exact on orthogonal proc faces); the non-orthogonal correction
 *     is not applied at processor faces (see FaceNormalGradient docs).
 *
 *  Marked final to enable devirtualisation of the kernel calls in GeometryScheme::update().
 */
class BasicGeometryScheme final : public GeometrySchemeFactory
{

public:

    BasicGeometryScheme(const UnstructuredMesh& mesh);

    void updateWeights(const Executor& exec, SurfaceField<scalar>& weights) final;

    void updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs) final;

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
