// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief
 *
 */
template<typename FieldValueType, typename AssemblyType = FieldValueType>
class GaussGreenDiv :
    public DivOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenDiv<FieldValueType, AssemblyType>>
{
    using Base = DivOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenDiv<FieldValueType, AssemblyType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs) {};

    virtual VolumeField<FieldValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(VolumeField<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(Vector<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    virtual void
    div(la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> clone() const override
    {
        return std::make_unique<GaussGreenDiv<FieldValueType, AssemblyType>>(*this);
    }

private:

    SurfaceInterpolation<FieldValueType> surfaceInterpolation_;
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation of table() static local, so the DLL's addSubType() inserts into
// a different map than the one the test binary queries.
extern template class GaussGreenDiv<scalar>;
extern template class GaussGreenDiv<Vec3>;
extern template class GaussGreenDiv<Vec3, scalar>;

/* @brief Adds the explicit (deferred) divergence correction of a corrected interpolation scheme
 * to the linear-system rhs. Shared with the fused div-laplacian operator so both implicit div
 * paths apply linearUpwind identically. Defined and instantiated in gaussGreenDiv.cpp.
 */
template<typename FieldValueType, typename AssemblyType>
void addDivCorrectionToRhs(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<FieldValueType>& correction,
    const dsl::Coeff operatorScaling
);

extern template void addDivCorrectionToRhs(
    la::LinearSystem<scalar, scalar>&,
    const SurfaceField<scalar>&,
    const SurfaceField<scalar>&,
    const dsl::Coeff
);
extern template void addDivCorrectionToRhs(
    la::LinearSystem<Vec3, Vec3>&,
    const SurfaceField<scalar>&,
    const SurfaceField<Vec3>&,
    const dsl::Coeff
);
extern template void addDivCorrectionToRhs(
    la::LinearSystem<scalar, Vec3>&,
    const SurfaceField<scalar>&,
    const SurfaceField<Vec3>&,
    const dsl::Coeff
);

} // namespace NeoN
