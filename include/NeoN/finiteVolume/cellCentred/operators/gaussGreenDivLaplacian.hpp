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

    void implicitOperation(la::LinearSystem<scalar, ValueType>& ls) const
        requires(!std::is_same_v<ValueType, scalar>);

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

    // True when the div scheme carried a leading `bounded` prefix. The fused kernel then also emits
    // the bounded-convection Sp diagonal term (applyBoundedDivDiagonal), matching the un-fused
    // BoundedDiv path -- without it the momentum matrix loses its boundedness stabilisation and the
    // solve diverges.
    bool bounded_ = false;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation, causing duplicate-symbol linker errors and bloating compile times.
extern template class GaussGreenDivLaplacian<scalar>;
extern template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
