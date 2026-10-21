// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// Shared kernel behind inletOutlet. Per boundary face it switches between two mixed-BC states
// according to the sign of the face flux phi:
//   inflow  (phi <  0): Dirichlet  -> valueFraction = 1, refValue = inletValue, refGrad = 0
//   outflow (phi >= 0): zeroGradient -> valueFraction = 0, refGrad = 0
// The explicit face value is the evaluated mixed value used by convection/interpolation:
//   value = valueFraction*inletValue + (1 - valueFraction)*(phi_C + refGrad/delta)
// with refGrad = 0 this is inletValue on inflow and the owner-cell value on outflow.
//
// When no flux is available (hasFlux == false, e.g. the plain correctBoundaryCondition path with
// no BoundaryContext) every face is treated as outflow, degenerating to a pure zeroGradient BC.
template<typename ValueType>
void setInletOutletValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    ValueType inletValue,
    View<const NeoN::scalar> boundaryFlux,
    bool hasFlux
)
{
    const auto internalV = domainVector.internalVector().view();

    auto [refGradV, valueV, valueFractionV, refValueV, boundaryFaceOwnersV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const auto internal = internalV[owner];

            // Inflow where the face flux is negative; otherwise outflow (zero-gradient).
            const NeoN::scalar valueFraction = (hasFlux && boundaryFlux[i] < 0.0) ? 1.0 : 0.0;

            valueFractionV[i] = valueFraction;
            refValueV[i] = inletValue;
            refGradV[i] = NeoN::zero<ValueType>();
            valueV[i] = inletValue * valueFraction + internal * (1.0 - valueFraction);
        },
        "setInletOutletValue"
    );
}

}

// inletOutlet switches between a fixed inlet value (on inflow faces) and zero-gradient
// extrapolation (on outflow faces), decided per face by the sign of the face flux phi. It is the
// generic-transport outlet counterpart to fixedValue and mirrors OpenFOAM's inletOutlet
// (a mixedFvPatchField with valueFraction = neg(phi), refGrad = 0).
//
// The flux is looked up by name (key "phi", default "phi") from the BoundaryContext passed to the
// context-aware correctBoundaryCondition overload. The plain overload has no flux and degenerates
// to zeroGradient, so a solver that wants the dynamic switch must call the field's
// correctBoundaryConditions(ctx) with phi inserted into the context.
template<typename ValueType>
class InletOutlet :
    public VolumeBoundaryFactory<ValueType>::template Register<InletOutlet<ValueType>>
{
    using Base =
        typename VolumeBoundaryFactory<ValueType>::template Register<InletOutlet<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    using InletOutletType = InletOutlet<ValueType>;

    // assignable = true: the boundary value is recomputed every correctBoundaryCondition call, so
    // any downstream overwrite is reversible. fixesValue = false: the per-face
    // refValue/valueFraction drive the operators via the mixed-BC kernels; a global Dirichlet flag
    // would over-constrain mixed-flow patches.
    InletOutlet(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh),
          inletValue_(dict.get<ValueType>("inletValue")),
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi"))
    {}

    // No context => no flux available: behave as zero-gradient (all faces treated as outflow).
    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setInletOutletValue(
            domainVector, mesh_, this->range(), inletValue_, View<const NeoN::scalar> {}, false
        );
    }

    // Context-aware: read the face flux phi and switch inflow/outflow per face.
    virtual void
    correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasSurfaceScalar(phiName_))
        {
            const auto& phi = ctx.surfaceScalarField(phiName_);
            detail::setInletOutletValue(
                domainVector,
                mesh_,
                this->range(),
                inletValue_,
                phi.boundaryData().value().view(),
                true
            );
        }
        else
        {
            correctBoundaryCondition(domainVector);
        }
    }

    static std::string name() { return "inletOutlet"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Inlet/outlet: fixed inletValue on inflow faces, zero-gradient on outflow faces "
               "(switched per face by the sign of the face flux phi).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<InletOutlet>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType inletValue_;
    std::string phiName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
