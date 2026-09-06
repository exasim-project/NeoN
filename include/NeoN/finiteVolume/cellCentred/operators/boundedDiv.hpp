// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Bounded convection div scheme wrapper.
 *
 * Mirrors OpenFOAM's Foam::fv::boundedConvectionScheme: wraps an inner div
 * scheme and adds an Sp-style correction proportional to the local
 * continuity error, -fvc::surfaceIntegrate(faceFlux) * psi (explicit) or
 * -fvm::Sp(fvc::surfaceIntegrate(faceFlux), psi) (implicit).
 *
 * Reads as `bounded <innerScheme...>`, e.g. `bounded Gauss upwind`.
 */
template<typename FieldValueType, typename AssemblyType = FieldValueType>
class BoundedDiv :
    public DivOperatorFactory<FieldValueType, AssemblyType>::template Register<
        BoundedDiv<FieldValueType, AssemblyType>>
{
    using Base = typename DivOperatorFactory<FieldValueType, AssemblyType>::template Register<
        BoundedDiv<FieldValueType, AssemblyType>>;

public:

    static std::string name() { return "bounded"; }

    static std::string doc()
    {
        return "Bounded convection scheme wrapper. Reads `bounded <inner>` "
               "and adds -fvm::Sp(surfaceIntegrate(faceFlux), psi) to the inner "
               "scheme's discretisation so continuity-error noise can't push "
               "ψ negative cell-to-cell.";
    }

    static std::string schema() { return "none"; }

    BoundedDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs);

    void
    div(VolumeField<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    void
    div(Vector<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    void
    div(la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    VolumeField<FieldValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const override;

    std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> clone() const override;

private:

    /* @brief Wraps an already-built inner div scheme, used by clone(). */
    BoundedDiv(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> inner
    );

    std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> inner_;
};

/* @brief Applies the bounded-convection Sp diagonal correction,
 * A[i,i] -= (sum of faceFlux over cell i) * scaling[i], over internal,
 * physical-boundary and processor-boundary faces to an already-assembled
 * linear system.
 */
template<typename FieldValueType, typename AssemblyType = FieldValueType>
void applyBoundedDivDiagonal(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
);

extern template class BoundedDiv<scalar>;
extern template class BoundedDiv<Vec3>;
extern template class BoundedDiv<Vec3, scalar>;

} // namespace NeoN::finiteVolume::cellCentred
