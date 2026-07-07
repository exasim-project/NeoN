// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief bounded explicit MULES (FCT) solve of a scalar transport equation.
 *
 * Reproduces OpenFOAM's MULES::explicitSolve/limit/limiter
 * (src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C) for the simplified
 * path used by the VoF alpha equation:
 *   rho == 1 (geometricOneField), Sp == Su == 0, static mesh,
 *   extremaCoeff == smoothLimiter == 0, no coupled/fixed-value boundary widening.
 *
 * On entry @p alphaPhi holds the high-order (unlimited) face flux and @p alpha holds
 * the current == old-time field. The FCT limiter blends @p alphaPhi towards the
 * bounded-donor (upwind) flux face-by-face so that the conservative explicit update
 *   alpha := alpha - deltaT * surfaceIntegrate(alphaPhi)
 * keeps psiMin <= alpha <= psiMax without any clamp. Both @p alpha and @p alphaPhi
 * are modified in place: @p alphaPhi becomes the limited flux, @p alpha the update.
 *
 * The donor (upwind) flux phiBD is computed inline from @p phi / @p alpha; on
 * boundary faces phiBD == alphaPhi (⇒ boundary phiCorr == 0), matching OpenFOAM's
 * non-coupled boundary handling. Reuses fvcc::surfaceIntegrate for the update.
 *
 * @param alpha        [in,out] scalar field (== oldTime at entry); advanced in place
 * @param phi          [in]     volumetric face flux
 * @param alphaPhi     [in,out] high-order flux in; limited flux out
 * @param deltaT       [in]     time-step size
 * @param psiMax       [in]     upper bound (default 1)
 * @param psiMin       [in]     lower bound (default 0)
 * @param nLimiterIter [in]     number of FCT limiter sweeps (default 3)
 */
void mulesExplicitSolve(
    VolumeField<scalar>& alpha,
    const SurfaceField<scalar>& phi,
    SurfaceField<scalar>& alphaPhi,
    scalar deltaT,
    scalar psiMax = 1.0,
    scalar psiMin = 0.0,
    int nLimiterIter = 3
);

} // namespace NeoN::finiteVolume::cellCentred
