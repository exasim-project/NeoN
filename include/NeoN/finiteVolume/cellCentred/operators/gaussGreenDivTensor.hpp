// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/dsl/coeff.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @brief Gauss-Green divergence of a tensor field: div(T) = (1/V) * Σ_f (S_f · T_f)
 *
 * Unlike the flux-weighted div(phi, field), this contracts face-interpolated tensors
 * with face area normals to produce a vector field.
 */
class GaussGreenDivTensor
{

public:

    GaussGreenDivTensor(const Executor& exec, const UnstructuredMesh& mesh);

    VolumeField<Vec3>
    div(const VolumeField<Tensor>& T, const dsl::Coeff operatorScaling = dsl::Coeff {}) const;

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    SurfaceInterpolation<Tensor> surfaceInterpolation_;
};

} // namespace NeoN::finiteVolume::cellCentred
