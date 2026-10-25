// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/fixedFluxPressure.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

/* @brief Gradient-energy boundary condition.
 *
 * Mirrors OpenFOAM's gradientEnergyFvPatchScalarField: the boundary GRADIENT (refGrad) is
 * set EXTERNALLY (by NeoN::finiteVolume::cellCentred::mapEnergyBoundaryFromTemperature, see
 * energyBoundaryMapping.hpp) every call from the current temperature boundary gradient
 * (chain-rule-mapped through Cv), so he's Neumann-type boundary (e.g. an adiabatic/
 * zero-heat-flux wall) tracks T's own gradient-type condition live. Reuses
 * detail::applyFixedFluxPressure -- the read-don't-overwrite "external push" mechanism it
 * implements is exactly what's needed here too, under a name that matches the physical
 * quantity (energy, not pressure) this boundary actually represents.
 */
template<typename ValueType>
class GradientEnergy :
    public VolumeBoundaryFactory<ValueType>::template Register<GradientEnergy<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<GradientEnergy<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    GradientEnergy(const UnstructuredMesh& mesh, const Dictionary&, localIdx patchID)
        : Base(mesh, {}, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::applyFixedFluxPressure(domainVector, mesh_, this->range());
    }

    static std::string name() { return "gradientEnergy"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Gradient-energy boundary condition (externally updatable per-face refGrad, "
               "mirrors OpenFOAM's gradientEnergyFvPatchScalarField).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<GradientEnergy>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

}
