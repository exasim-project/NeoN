// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
void computeLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const FaceNormalGradient<ValueType>& faceNormalGradient
);

template<typename ValueType>
void computeLaplacianProcBoundImpl(
    la::LinearSystem<ValueType>& ls,
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
        const dsl::Coeff coeff
    ) override;

    virtual VolumeField<ValueType> laplacian(
        const SurfaceField<scalar>& gamma, const VolumeField<ValueType>& phi, const dsl::Coeff coeff
    ) const override;

    virtual void laplacian(
        Vector<ValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    virtual void laplacian(
        la::LinearSystem<ValueType>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    std::unique_ptr<LaplacianOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<ValueType>>(*this);
    };

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;

    FaceNormalGradient<ValueType> faceNormalGradient_;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation of table() static local, so the DLL's addSubType() inserts into
// a different map than the one the test binary queries.
extern template class GaussGreenLaplacian<scalar>;
extern template class GaussGreenLaplacian<Vec3>;

} // namespace NeoN
