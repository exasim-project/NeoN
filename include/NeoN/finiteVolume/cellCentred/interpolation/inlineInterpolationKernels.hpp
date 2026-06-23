// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief Device-callable weight kernel for linear interpolation.
** Captures the pre-computed geometry weights views (internal + boundary); no virtual dispatch.
** The boundary view covers both physical and proc boundary faces in a single contiguous array.
*/
struct LinearInlineKernel
{
    View<const scalar> weights;
    View<const scalar> bWeights;

    NEON_INLINE_FUNCTION scalar weight(localIdx facei, scalar /*flux*/) const
    {
        return weights[facei];
    }

    NEON_INLINE_FUNCTION scalar boundaryWeight(localIdx bfacei, scalar /*flux*/) const
    {
        return bWeights[bfacei];
    }

    NEON_INLINE_FUNCTION scalar procBoundaryWeight(localIdx bcfacei, scalar /*flux*/) const
    {
        return bWeights[bcfacei];
    }
};

/* @brief Device-callable weight kernel for upwind interpolation.
** Internal weight: 1 if flux >= 0 (owner upwind), 0 otherwise.
** Physical boundary weight: always 1 (value comes from the BC, not interpolated).
** Proc boundary weight: flux-sign, matching the coupled internal-face convention.
*/
struct UpwindInlineKernel
{
    NEON_INLINE_FUNCTION scalar weight([[maybe_unused]] localIdx facei, scalar flux) const
    {
        return flux >= scalar(0) ? scalar(1) : scalar(0);
    }

    NEON_INLINE_FUNCTION scalar
    boundaryWeight([[maybe_unused]] localIdx bfacei, [[maybe_unused]] scalar flux) const
    {
        return scalar(1);
    }

    NEON_INLINE_FUNCTION scalar
    procBoundaryWeight([[maybe_unused]] localIdx bcfacei, scalar flux) const
    {
        return flux >= scalar(0) ? scalar(1) : scalar(0);
    }
};

using InlineWeightKernel = std::variant<LinearInlineKernel, UpwindInlineKernel>;

} // namespace NeoN::finiteVolume::cellCentred
