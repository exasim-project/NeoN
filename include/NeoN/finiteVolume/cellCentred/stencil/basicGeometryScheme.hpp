// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
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
        const SurfaceField<scalar>& nonOrthDeltaCoeffs,
        SurfaceField<Vec3>& nonOrthCorrectionVec3s
    ) final;


private:

    const UnstructuredMesh& mesh_;

#ifdef NF_WITH_MPI_SUPPORT
    /// Neighbour-owner cell centre (global coords) per processor face, exchanged once per geometry
    /// update and reused by all three update kernels, so a full update performs a single proc-face
    /// halo exchange instead of one per kernel. Refreshed by updateWeights (the first kernel
    /// GeometryScheme::update() runs); updateNonOrthDeltaCoeffs and updateNonOrthCorrectionVec3s
    /// read it. nullopt until the first updateWeights; size 0 on serial / interior-only partitions.
    std::optional<Vector<Vec3>> procNeiCellCentre_;
#endif
};

} // namespace NeoN
