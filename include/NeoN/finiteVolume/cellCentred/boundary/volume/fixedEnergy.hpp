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
// Recompute the boundary value from the EXISTING per-face refValue (does NOT overwrite
// refValue with a stored uniform, unlike FixedValue). This is the fixedEnergy semantics,
// mirroring OpenFOAM's fixedEnergyFvPatchScalarField: an external caller
// (mapEnergyBoundaryFromTemperature) sets refValue every call from the current temperature
// boundary value, so correctBoundaryCondition only re-derives the boundary value from
// whatever refValue currently holds.
template<typename ValueType>
void applyFixedEnergy(Field<ValueType>& domainVector, std::pair<localIdx, localIdx> range)
{
    auto [value, valueFraction, refValue, refGradient] = views(
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        domainVector.boundaryData().refGrad()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            // reads (does not write) refValue[i], set externally
            value[i] = refValue[i];
            valueFraction[i] = 1.0;             // only use refValue
            refGradient[i] = zero<ValueType>(); // not used
        },
        "applyFixedEnergy"
    );
}
}

/* @brief Fixed-energy boundary condition.
 *
 * Mirrors OpenFOAM's fixedEnergyFvPatchScalarField: the boundary VALUE (refValue) is set
 * EXTERNALLY (by NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature, see
 * energyBoundaryMapping.hpp) every call from the current temperature boundary value, so
 * he's fixedValue-type boundary (e.g. a fixed-temperature inlet) tracks T live rather than
 * being frozen at whatever he happened to be at construction. correctBoundaryCondition only
 * recomputes the boundary value from the current refValue, never clobbering it with a stored
 * uniform. refValue is zero-initialised by the boundary-data allocation, so the BC behaves as
 * a zero fixedValue until the first mapEnergyBoundaryFromTemperature call.
 */
template<typename ValueType>
class FixedEnergy :
    public VolumeBoundaryFactory<ValueType>::template Register<FixedEnergy<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<FixedEnergy<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    FixedEnergy(const UnstructuredMesh& mesh, const Dictionary&, localIdx patchID)
        : Base(mesh, {}, patchID, {.assignable = true, .fixesValue = true})
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::applyFixedEnergy(domainVector, this->range());
    }

    static std::string name() { return "fixedEnergy"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Fixed-energy boundary condition (externally updatable per-face refValue, "
               "mirrors OpenFOAM's fixedEnergyFvPatchScalarField).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<FixedEnergy>(*this);
    }
};

}
