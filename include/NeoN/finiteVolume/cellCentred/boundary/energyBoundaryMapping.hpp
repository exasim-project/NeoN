// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/boundaryData.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Map a temperature field's boundary data onto an energy field's boundary data
 * through the constant-specific-heat relation `he = Cv * (T - Tref)`.
 *
 * Mirrors what OpenFOAM's basicThermo does internally for every solver that transports `he`
 * rather than `T` directly (fixedEnergyFvPatchScalarField/gradientEnergyFvPatchScalarField,
 * src/thermophysicalModels/basic/derivedFvPatchFields/): a solver that reads `0/T` BCs but
 * transports `he` calls this once per outer iteration (after `T.correctBoundaryConditions()`,
 * before assembling/solving the energy equation) to push the current `T` boundary state into
 * `he`'s own boundary storage, then calls `he.correctBoundaryConditions()` so each patch's own
 * BC type (FixedEnergy/GradientEnergy, see fixedEnergy.hpp/gradientEnergy.hpp) re-derives
 * `he`'s boundary value from what was just written.
 *
 * `he = Cv*(T - Tref)` is affine, so the map is applied component-wise to both halves of the
 * mixed-BC representation every NeoN boundary condition shares (refValue/refGrad/
 * valueFraction, see e.g. fixedFluxPressure.hpp):
 *   - refValue (the Dirichlet/fixedValue-type target):  he.refValue = Cv*(T.refValue - Tref)
 *   - refGrad  (the Neumann/gradient-type target):      he.refGrad  = Cv*T.refGrad
 *   - valueFraction (the Dirichlet/Neumann blend): copied through unchanged -- the blend
 *     itself doesn't depend on which units are being transported.
 * This correctly reproduces OpenFOAM's fixedEnergy BC exactly (a pure Dirichlet target) and
 * its gradientEnergy BC exactly for constant, spatially-uniform Cv (where the general
 * chain-rule gradient map collapses to this simple scaling); a spatially-varying or
 * temperature-dependent Cv would need the per-face Cv as a Field rather than a scalar here --
 * out of scope for this constant-Cv implementation.
 *
 * Internal field values are untouched entirely -- this operates on boundary data only (it
 * never needed a whole Field, just the refValue/refGrad/valueFraction triple), so it takes
 * BoundaryData directly: call as
 * `mapEnergyBoundaryFromTemperature(T.boundaryData(), he.boundaryData(), Cv, Tref)`. Call
 * `he.correctBoundaryConditions()` afterward to materialise the values.
 */
template<typename ValueType>
void mapEnergyBoundaryFromTemperature(
    const BoundaryData<ValueType>& temperature,
    BoundaryData<ValueType>& energy,
    scalar Cv,
    scalar Tref
)
{
    auto [heRefValue, heRefGrad, heValueFraction] =
        views(energy.refValue(), energy.refGrad(), energy.valueFraction());
    auto [TRefValue, TRefGrad, TValueFraction] =
        views(temperature.refValue(), temperature.refGrad(), temperature.valueFraction());

    NeoN::parallelFor(
        energy.exec(),
        {0, energy.refValue().size()},
        NEON_LAMBDA(const localIdx i) {
            heRefValue[i] = Cv * (TRefValue[i] - Tref);
            heRefGrad[i] = Cv * TRefGrad[i];
            heValueFraction[i] = TValueFraction[i];
        },
        "mapEnergyBoundaryFromTemperature"
    );
}

}
