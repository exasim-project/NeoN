// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @brief Calculates courant number from the face fluxes.
 * @param faceFlux Scalar surface field with the flux values of all faces.
 * @param dt Size of the time step.
 * @return Maximum courant number.
 */
NeoFOAM::scalar computeCoNum(const SurfaceField<NeoFOAM::scalar>& faceFlux, const scalar dt);

} // namespace NeoFOAM
