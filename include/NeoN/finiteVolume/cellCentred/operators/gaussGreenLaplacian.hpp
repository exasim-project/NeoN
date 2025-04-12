// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLaplacianExp(
    const FaceNormalGradient<ValueType>&,
    const SurfaceField<scalar>&,
    VolumeField<ValueType>&,
    Field<ValueType>&,
    const dsl::Coeff
);

template<typename ValueType>
void computeLaplacianImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& gamma,
    VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const SparsityPattern& sparsityPattern,
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
          faceNormalGradient_(exec, mesh, inputs),
          sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};

    virtual void laplacian(
        VolumeField<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianExp<ValueType>(
            faceNormalGradient_, gamma, phi, lapPhi.internalField(), operatorScaling
        );
    };

    virtual void laplacian(
        Field<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianExp<ValueType>(faceNormalGradient_, gamma, phi, lapPhi, operatorScaling);
    };

    virtual void laplacian(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) override
    {
        computeLaplacianImpl(
            ls, gamma, phi, operatorScaling, *sparsityPattern_.get(), faceNormalGradient_
        );
    };

    std::unique_ptr<LaplacianOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<ValueType>>(*this);
    };

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;

    FaceNormalGradient<ValueType> faceNormalGradient_;

    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

// instantiate the template class
template class GaussGreenLaplacian<scalar>;
template class GaussGreenLaplacian<Vector>;

} // namespace NeoFOAM
