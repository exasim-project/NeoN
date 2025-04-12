// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Calculates courant number from the face fluxes.
 * @param faceFlux Scalar surface field with the flux values of all faces.
 * @param dt Size of the time step.
 * @return Maximum courant number.
 */
NeoN::scalar computeCoNum(const SurfaceField<NeoN::scalar>& faceFlux, const scalar dt);

} // namespace NeoN
