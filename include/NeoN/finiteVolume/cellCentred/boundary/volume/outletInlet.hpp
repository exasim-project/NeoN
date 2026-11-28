// SPDX-FileCopyrightText: 2026 NeoN authors
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
// Kernel behind outletInlet, the mirror image of inletOutlet. Per boundary face it switches
// between two mixed-BC states according to the sign of the face flux phi:
//   outflow (phi >  0): Dirichlet    -> valueFraction = 1, refValue = outletValue, refGrad = 0
//   inflow  (phi <= 0): zeroGradient -> valueFraction = 0, refGrad = 0
// as OpenFOAM's outletInlet (a mixedFvPatchField with valueFraction = pos0(phi) evaluated on the
// patch flux; OpenFOAM uses pos0, i.e. phi = 0 counts as outflow, which is reproduced here).
// The explicit face value is
//   value = valueFraction*outletValue + (1 - valueFraction)*phi_C.
//
// When no flux is available (hasFlux == false) every face is treated as inflow, degenerating to
// a pure zeroGradient BC (the same fallback as inletOutlet).
template<typename ValueType>
void setOutletInletValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    ValueType outletValue,
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

            // Outflow where the face flux is non-negative; otherwise inflow (zero-gradient).
            const NeoN::scalar valueFraction = (hasFlux && boundaryFlux[i] >= 0.0) ? 1.0 : 0.0;

            valueFractionV[i] = valueFraction;
            refValueV[i] = outletValue;
            refGradV[i] = NeoN::zero<ValueType>();
            valueV[i] = outletValue * valueFraction + internal * (1.0 - valueFraction);
        },
        "setOutletInletValue"
    );
}

}

// outletInlet fixes the value on outflow faces and extrapolates (zero gradient) on inflow faces,
// decided per face by the sign of the face flux phi. It mirrors OpenFOAM's outletInlet
// (mixedFvPatchField with valueFraction = pos0(phi), refValue = outletValue, refGrad = 0).
//
// The flux is looked up by name (key "phi", default "phi") from the BoundaryContext passed to the
// context-aware correctBoundaryCondition overload. The plain overload has no flux and degenerates
// to zeroGradient.
template<typename ValueType>
class OutletInlet :
    public VolumeBoundaryFactory<ValueType>::template Register<OutletInlet<ValueType>>
{
    using Base =
        typename VolumeBoundaryFactory<ValueType>::template Register<OutletInlet<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    // As inletOutlet: the boundary value is recomputed every correction (assignable), and the
    // per-face refValue/valueFraction drive the operators (no global Dirichlet flag).
    OutletInlet(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh),
          outletValue_(dict.get<ValueType>("outletValue")),
          phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi"))
    {}

    // No context => no flux available: behave as zero-gradient.
    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setOutletInletValue(
            domainVector, mesh_, this->range(), outletValue_, View<const NeoN::scalar> {}, false
        );
    }

    // Context-aware: read the face flux phi and switch outflow/inflow per face.
    virtual void
    correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasSurfaceScalar(phiName_))
        {
            const auto& phi = ctx.surfaceScalarField(phiName_);
            detail::setOutletInletValue(
                domainVector,
                mesh_,
                this->range(),
                outletValue_,
                phi.boundaryData().value().view(),
                true
            );
        }
        else
        {
            correctBoundaryCondition(domainVector);
        }
    }

    static std::string name() { return "outletInlet"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Outlet/inlet: fixed outletValue on outflow faces, zero-gradient on inflow faces "
               "(switched per face by the sign of the face flux phi).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<OutletInlet>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType outletValue_;
    std::string phiName_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
