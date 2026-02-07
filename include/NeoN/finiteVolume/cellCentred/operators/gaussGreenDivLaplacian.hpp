// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
void computeDivLapImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    const la::SparsityPattern& sp
);

/* @brief
 *
 */
template<typename ValueType>
class GaussGreenDivLaplacian :
    public DivOperatorFactory<ValueType>::template Register<GaussGreenDivLaplacian<ValueType>>
{
    using Base =
        DivOperatorFactory<ValueType>::template Register<GaussGreenDivLaplacian<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDivLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs) {};

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
        VolumeField<ValueType> divPhi(
            this->exec_,
            name,
            this->mesh_,
            createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
        );
        NeoN::fill(divPhi.internalVector(), zero<ValueType>());
        NeoN::fill(divPhi.boundaryData().value(), zero<ValueType>());
        // computeDivExp<ValueType>(
        //     faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
        // );
        return divPhi;
    };

    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        // computeDivExp<ValueType>(
        //     faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
        // );
    }

    virtual void
    div(Vector<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override {
        // computeDivExp<ValueType>(faceFlux, phi, surfaceInterpolation_, divPhi, operatorScaling);
    };

    virtual void
    div(la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override {
        // computeDivImp(
        //     ls, faceFlux, phi, surfaceInterpolation_, operatorScaling, this->getSparsityPattern()
        // );
    };

    std::unique_ptr<DivOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenDivLaplacian<ValueType>>(*this);
    }

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;
};

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
