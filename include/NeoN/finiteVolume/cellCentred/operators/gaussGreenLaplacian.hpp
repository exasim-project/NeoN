// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>&,
    const SurfaceField<scalar>&,
    const VolumeField<ValueType>&,
    Vector<ValueType>&,
    const dsl::Coeff
);

template<typename ValueType>
void computeLaplacianImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
);

template<typename ValueType>
class GaussGreenLaplacian :
    public LaplacianOperatorFactory<ValueType>::template Register<GaussGreenLaplacian<ValueType>>
{
    using Base =
        LaplacianOperatorFactory<ValueType>::template Register<GaussGreenLaplacian<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Laplacian"; }

    static std::string schema() { return "none"; }

    GaussGreenLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs) {};

    virtual void laplacian(
        VolumeField<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianExp<ValueType>(
            faceNormalGradient_, gamma, phi, lapPhi.internalVector(), operatorScaling
        );
    };

    virtual VolumeField<ValueType> laplacian(
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) const override
    {
        std::string name = "laplacian(" + gamma.name + "," + phi.name + ")";
        VolumeField<ValueType> lapPhi(
            this->exec_,
            name,
            this->mesh_,
            createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
        );
        NeoN::fill(lapPhi.internalVector(), zero<ValueType>());
        NeoN::fill(lapPhi.boundaryData().value(), zero<ValueType>());
        computeLaplacianExp<ValueType>(
            faceNormalGradient_, gamma, phi, lapPhi.internalVector(), operatorScaling
        );
        return lapPhi;
    };

    virtual void laplacian(
        Vector<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianExp<ValueType>(faceNormalGradient_, gamma, phi, lapPhi, operatorScaling);
    };

    virtual void laplacian(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianImpl(ls, gamma, phi, operatorScaling, faceNormalGradient_);
    };

    std::unique_ptr<LaplacianOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<ValueType>>(*this);
    };

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;

    FaceNormalGradient<ValueType> faceNormalGradient_;
};

// instantiate the template class
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vec3>;

} // namespace NeoN
