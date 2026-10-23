// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Shared kernel behind SineFixedValue, a time-varying Dirichlet BC mirroring OpenFOAM's
// uniformFixedValue patch field when its Function1 is a sine (Foam::Function1Types::Sine, "sin"
// form -- the default waveType). Per OpenFOAM's Sine::sinValue:
//   value(t) = amplitude * sin(2*pi * frequency * (t - t0)) * scale + level
// (amplitude defaults to 1, t0 defaults to 0, matching Sine's own dict-read defaults). The whole
// patch gets one uniform value every call (valueFraction = 1, refGrad = 0), recomputed from the
// current simulation time -- mirroring uniformFixedValueFvPatchField::updateCoeffs(), which
// re-evaluates its Function1 at db().time().timeOutputValue() on every call rather than caching
// the value.
//
// When no time is available (haveContext == false, e.g. the plain correctBoundaryCondition path
// with no BoundaryContext) the patch degrades to zero-gradient, the same fallback convention
// every other context-aware NeoN BC uses on missing context.
inline void setSineFixedValue(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::scalar amplitude,
    NeoN::scalar frequency,
    NeoN::scalar scale,
    NeoN::scalar level,
    NeoN::scalar t0,
    NeoN::scalar time,
    bool haveContext
)
{
    const auto internalV = domainVector.internalVector().view();
    const auto boundaryFaceOwnersV = mesh.boundaryMesh().faceOwners().view();

    auto [refGradV, valueV, valueFractionV, refValueV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::scalar uniformValue = NeoN::scalar(0);
    if (haveContext)
    {
        constexpr NeoN::scalar twoPi = NeoN::scalar(2) * Kokkos::numbers::pi_v<NeoN::scalar>;
        uniformValue = amplitude * Kokkos::sin(twoPi * frequency * (time - t0)) * scale + level;
    }

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradV[i] = NeoN::scalar(0);

            if (!haveContext)
            {
                const localIdx owner = boundaryFaceOwnersV[i];
                const NeoN::scalar ownerValue = internalV[owner];
                valueFractionV[i] = NeoN::scalar(0);
                refValueV[i] = ownerValue;
                valueV[i] = ownerValue;
                return;
            }

            valueFractionV[i] = NeoN::scalar(1);
            refValueV[i] = uniformValue;
            valueV[i] = uniformValue;
        },
        "setSineFixedValue"
    );
}

} // namespace detail

// SineFixedValue: a time-varying Dirichlet BC mirroring OpenFOAM's uniformFixedValue patch field
// configured with a sine Function1. See detail::setSineFixedValue for the exact formula.
//
// Needs, via the context-aware correctBoundaryCondition(domainVector, ctx) overload, the current
// simulation time (BoundaryContext::insertValue, key "time"). Without it, degrades to a plain
// zero-gradient extrapolation (the same fallback the plain, context-less overload always uses).
class SineFixedValue : public VolumeBoundaryFactory<NeoN::scalar>::template Register<SineFixedValue>
{
    using Base = VolumeBoundaryFactory<NeoN::scalar>::template Register<SineFixedValue>;

public:

    using Base::correctBoundaryCondition;

    // assignable = true: recomputed every correctBoundaryConditions() call. fixesValue = true:
    // on inflow this is a genuine global Dirichlet patch, same as fixedValue -- unlike the mixed
    // BCs elsewhere in this file, there is no flow-dependent switch to a Neumann branch.
    SineFixedValue(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = true}), mesh_(mesh),
          amplitude_(
              dict.contains("amplitude") ? dict.get<NeoN::scalar>("amplitude") : NeoN::scalar(1)
          ),
          frequency_(dict.get<NeoN::scalar>("frequency")), // required: no physically-sane default
          scale_(dict.get<NeoN::scalar>("scale")),         // required: no physically-sane default
          level_(dict.get<NeoN::scalar>("level")),         // required: no physically-sane default
          t0_(dict.contains("t0") ? dict.get<NeoN::scalar>("t0") : NeoN::scalar(0))
    {}

    // No context => no simulation time available: degrade to zero-gradient.
    virtual void correctBoundaryCondition(Field<NeoN::scalar>& domainVector) final
    {
        detail::setSineFixedValue(
            domainVector,
            mesh_,
            this->range(),
            amplitude_,
            frequency_,
            scale_,
            level_,
            t0_,
            NeoN::scalar(0),
            false
        );
    }

    // Context-aware: evaluate the sine at the current simulation time.
    virtual void
    correctBoundaryCondition(Field<NeoN::scalar>& domainVector, const BoundaryContext& ctx) final
    {
        if (!ctx.hasValue("time"))
        {
            correctBoundaryCondition(domainVector);
            return;
        }

        detail::setSineFixedValue(
            domainVector,
            mesh_,
            this->range(),
            amplitude_,
            frequency_,
            scale_,
            level_,
            t0_,
            ctx.value("time"),
            true
        );
    }

    static std::string name() { return "sineFixedValue"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Time-varying uniform Dirichlet value (OpenFOAM's uniformFixedValue with a sine "
               "Function1): value(t) = amplitude*sin(2*pi*frequency*(t - t0))*scale + level "
               "(amplitude defaults to 1, t0 defaults to 0). Needs the current simulation time "
               "in the BoundaryContext (key \"time\"); without it, degrades to zero-gradient.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::scalar>> clone() const final
    {
        return std::make_unique<SineFixedValue>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::scalar amplitude_;
    NeoN::scalar frequency_;
    NeoN::scalar scale_;
    NeoN::scalar level_;
    NeoN::scalar t0_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
