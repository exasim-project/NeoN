// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Calculates courant number from the face fluxes.
 * @param faceFlux Scalar surface field with the flux values of all faces.
 * @param dt Size of the time step.
 * @return Maximum and Mean courant number.
 */
std::pair<scalar, scalar> computeCoNum(const SurfaceField<scalar>& faceFlux, const scalar dt);

} // namespace NeoN
