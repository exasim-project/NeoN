// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
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

    // Concrete ELL overload -- covers both cases SpatialOperator's ELL dispatch ever actually
    // calls with, since ELLMatrix<scalar, localIdx> is the only ELL matrix type currently
    // dispatched by SpatialOperator: ValueType == scalar substitutes to the same-type ELL
    // signature, ValueType == Vec3 to the segregated one. Being non-template (unlike the old
    // implicitOperation<SystemMatrixType>), it's covered by the explicit class instantiation
    // below, so it can be declared here and defined in gaussGreenDivLaplacian.cpp.
    void implicitOperation(la::LinearSystem<scalar, ValueType, la::ELLMatrix<scalar, localIdx>>& ls
    ) const;

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

    // Shared by all three implicitOperation overloads above (all defined in
    // gaussGreenDivLaplacian.cpp), one source of truth for the assembly sequence instead of
    // duplicating it per format. Defined in the .cpp, not header-inline -- see the comment there.
    template<typename AssemblyType, typename SystemMatrixType>
    void implicitOperationImpl(la::LinearSystem<AssemblyType, ValueType, SystemMatrixType>& ls
    ) const;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation, causing duplicate-symbol linker errors and bloating compile times.
extern template class GaussGreenDivLaplacian<scalar>;
extern template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
