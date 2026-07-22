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

// Mirrors OpenFOAM's Foam::fv::boundedConvectionScheme
// (src/finiteVolume/finiteVolume/convectionSchemes/boundedConvectionScheme/
//  boundedConvectionScheme.C):
//
//   fvmDiv(F, ψ) = base.fvmDiv(F, ψ) - fvm::Sp(fvc::surfaceIntegrate(F), ψ)
//   fvcDiv(F, ψ) = base.fvcDiv(F, ψ) - fvc::surfaceIntegrate(F) * ψ
//
// The correction vanishes when ∇·F = 0 (continuity satisfied), and otherwise
// adds an Sp-style implicit source proportional to the local continuity error.
// On positive scalars (k, ω, …) this enforces an M-matrix-like sign pattern so
// numerical noise in the face flux can't drive ψ negative cell-to-cell.
//
// Reads as `bounded <innerScheme...>`, e.g. `bounded Gauss upwind`. The
// constructor consumes nothing extra of its own — the leading "bounded" was
// already consumed by DivOperatorFactory::create() — and forwards the
// remaining tokens to a recursive create() to build the inner scheme.
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

    // Used by clone(): wraps an already-built inner factory without
    // re-parsing tokens. Public-via-friend so clone() can reach it.
    BoundedDiv(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> inner
    );

    std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> inner_;
};

// Applies the bounded-convection Sp diagonal correction
//   A[i,i] -= (Σ_f φ_f over cell i) * scaling[i]
// over internal, physical-boundary and processor-boundary faces to an already-assembled linear
// system. Factored out of BoundedDiv::div so the fused GaussGreenDivLaplacian can emit the SAME
// bounding term when its div scheme carries a leading `bounded` (a single definition also keeps the
// device kernel names unique across TUs).
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
