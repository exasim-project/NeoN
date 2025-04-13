// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Calculates courant number from the face fluxes.
 * @param faceFlux Scalar surface field with the flux values of all faces.
 * @param dt Size of the time step.
 * @return Maximum courant number.
 */
scalar computeCoNum(const SurfaceField<scalar>& faceFlux, const scalar dt);

} // namespace NeoN
