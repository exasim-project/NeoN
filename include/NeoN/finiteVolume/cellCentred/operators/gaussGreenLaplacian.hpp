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

// Deferred non-orthogonal correction kernel for the Laplacian (defined in gaussGreenLaplacian.cpp).
// Declared here so other operators (e.g. GaussGreenDivLaplacian) can reuse it. Adds the explicit
// snGrad correction to the linear system's rhs; a no-op unless the snGrad scheme is corrected.
template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeLaplacianNonOrthCorrImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
);

template<typename FieldValueType, typename AssemblyType = FieldValueType>
class GaussGreenLaplacian :
    public LaplacianOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenLaplacian<FieldValueType, AssemblyType>>
{
    using Base = LaplacianOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenLaplacian<FieldValueType, AssemblyType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Laplacian"; }

    static std::string schema() { return "none"; }

    GaussGreenLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs) {};

    virtual void laplacian(
        VolumeField<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    virtual VolumeField<FieldValueType> laplacian(
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) const override;

    virtual void laplacian(
        Vector<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    virtual void laplacian(
        la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    std::unique_ptr<LaplacianOperatorFactory<FieldValueType, AssemblyType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<FieldValueType, AssemblyType>>(*this);
    };

private:

    SurfaceInterpolation<FieldValueType> surfaceInterpolation_;

    FaceNormalGradient<FieldValueType> faceNormalGradient_;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation of table() static local, so the DLL's addSubType() inserts into
// a different map than the one the test binary queries.
extern template class GaussGreenLaplacian<scalar>;
extern template class GaussGreenLaplacian<Vec3>;
extern template class GaussGreenLaplacian<Vec3, scalar>;

} // namespace NeoN
