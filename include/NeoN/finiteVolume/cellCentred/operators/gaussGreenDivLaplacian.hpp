// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDivLaplacianIntImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

template<typename ValueType>
void computeDivLaplacianBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

template<typename ValueType>
void computeDivLaplacianProcBoundImpl(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

/* @brief
 *
 */
template<typename ValueType>
class GaussGreenDivLaplacian : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    GaussGreenDivLaplacian(const Executor& exec, Dictionary divConfig, Dictionary lapConfig);

    void explicitOperation(Vector<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType>& ls) const;

    void read(const Input& input);

    std::string getName() const;

    Dictionary getConfig() const;

private:

    dsl::Coeff coeffA_; // div coeff
    dsl::Coeff coeffB_; // lap coeff

    const SurfaceField<scalar>& gamma_;
    const SurfaceField<scalar>& flux_;

    std::shared_ptr<SurfaceInterpolation<ValueType>> divSurfaceInterpolation_;
    std::shared_ptr<FaceNormalGradient<ValueType>> faceNormalGradient_;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation, causing duplicate-symbol linker errors and bloating compile times.
extern template class GaussGreenDivLaplacian<scalar>;
extern template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
