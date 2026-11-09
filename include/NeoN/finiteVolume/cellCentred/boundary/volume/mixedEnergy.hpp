// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
template<typename ValueType>
void applyMixedEnergy(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto iVector = domainVector.internalVector().view();

    auto [refGradient, value, valueFraction, refValue, boundaryFaceOwners, deltaCoeffs] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            // reads (does not write) refValue, refGrad and valueFraction, set externally
            const auto gradientValue =
                iVector[boundaryFaceOwners[i]] + refGradient[i] * (1 / deltaCoeffs[i]);
            value[i] = valueFraction[i] * refValue[i] + (1 - valueFraction[i]) * gradientValue;
        },
        "applyMixedEnergy"
    );
}
}

/* @brief Mixed-energy boundary condition.
 *
 * Mirrors OpenFOAM's mixedEnergyFvPatchScalarField, which heThermo derives for a temperature
 * patch that is itself mixed (inletOutlet, outletInlet, freestream, ...). refValue, refGrad and
 * valueFraction are set EXTERNALLY (by
 * NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature, see energyBoundaryMapping.hpp)
 * every call from the current temperature boundary, and correctBoundaryCondition only evaluates
 *   value = valueFraction*refValue + (1 - valueFraction)*(he_C + refGrad/delta),
 * so he follows T's inflow (fixed value) and outflow (zero gradient) switching live. A
 * gradientEnergy patch would discard the valueFraction and treat every face as gradient-type.
 */
template<typename ValueType>
class MixedEnergy :
    public VolumeBoundaryFactory<ValueType>::template Register<MixedEnergy<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<MixedEnergy<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    MixedEnergy(const UnstructuredMesh& mesh, const Dictionary&, localIdx patchID)
        : Base(mesh, {}, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::applyMixedEnergy(domainVector, mesh_, this->range());
    }

    static std::string name() { return "mixedEnergy"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Mixed-energy boundary condition (externally updatable per-face refValue, refGrad "
               "and valueFraction, mirrors OpenFOAM's mixedEnergyFvPatchScalarField).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<MixedEnergy>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
