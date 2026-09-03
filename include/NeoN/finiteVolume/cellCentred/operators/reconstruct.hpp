// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief explicit reconstruction of a cell vector field from a surface scalar flux.
 *
 * Computes the least-squares cell vector whose face projections reproduce the flux:
 *   reconstruct(ssf)_C = inv( Σ_f Sf⊗Sf / |Sf| ) & Σ_f (Sf / |Sf|) · ssf_f
 * with surfaceSum semantics — each internal face contributes to both its owner and
 * neighbour cell, each boundary face contributes to its owner cell. The per-cell
 * symmetric 3×3 matrix is inverted analytically.
 *
 * @param ssf [in] - surface scalar flux to reconstruct
 * @return the reconstructed cell-centred vector field (calculated BCs, corrected)
 */
VolumeField<Vec3> reconstruct(const SurfaceField<scalar>& ssf);

} // namespace NeoN::finiteVolume::cellCentred
