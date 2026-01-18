// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/convDiffOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Explicit fused conv+diff
template<typename ValueType>
void computeConvDiffExp(
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    Vector<ValueType>& result,
    const dsl::Coeff operatorScaling
);

// Implicit fused conv+diff
template<typename ValueType>
void computeConvDiffImp(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff operatorScaling,
    const la::SparsityPattern& sparsityPattern
);


/* @brief Gauss–Green fused convection–diffusion operator
 *
 * Combines GaussGreenDiv + GaussGreenLaplacian into a single face loop.
 */
template<typename ValueType>
class GaussGreenConvDiff :
    public ConvDiffOperatorFactory<ValueType>::template Register<GaussGreenConvDiff<ValueType>>
{
    using Base =
        ConvDiffOperatorFactory<ValueType>::template Register<GaussGreenConvDiff<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green fused Div + Laplacian"; }

    static std::string schema() { return "none"; }

    GaussGreenConvDiff(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs)
    {}

    // explicit into VolumeField
    void convDiff(
        VolumeField<ValueType>& result,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeConvDiffExp<ValueType>(
            faceFlux,
            gamma,
            phi,
            surfaceInterpolation_,
            faceNormalGradient_,
            result.internalVector(),
            operatorScaling
        );
    }

    // explicit return-by-value
    VolumeField<ValueType> convDiff(
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) const override
    {
        std::string name = "convDiff(" + faceFlux.name + "," + gamma.name + "," + phi.name + ")";
        VolumeField<ValueType> result(
            this->exec_,
            name,
            this->mesh_,
            createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
        );
        NeoN::fill(result.internalVector(), zero<ValueType>());
        NeoN::fill(result.boundaryData().value(), zero<ValueType>());

        computeConvDiffExp<ValueType>(
            faceFlux,
            gamma,
            phi,
            surfaceInterpolation_,
            faceNormalGradient_,
            result.internalVector(),
            operatorScaling
        );
        return result;
    }

    // explicit into Vector
    void convDiff(
        Vector<ValueType>& result,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeConvDiffExp<ValueType>(
            faceFlux,
            gamma,
            phi,
            surfaceInterpolation_,
            faceNormalGradient_,
            result,
            operatorScaling
        );
    }

    // implicit: build matrix
    void convDiff(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeConvDiffImp<ValueType>(
            ls,
            faceFlux,
            gamma,
            phi,
            surfaceInterpolation_,
            faceNormalGradient_,
            operatorScaling,
            this->getSparsityPattern()
        );
    }

    std::unique_ptr<ConvDiffOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenConvDiff<ValueType>>(*this);
    }

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;
    FaceNormalGradient<ValueType> faceNormalGradient_;
};

// explicit template instantiations
template class GaussGreenConvDiff<scalar>;
template class GaussGreenConvDiff<Vec3>;

} // namespace NeoN::finiteVolume::cellCentred
