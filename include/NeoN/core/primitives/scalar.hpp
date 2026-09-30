// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/portability.hpp"

#include "NeoN/core/primitives/traits.hpp"

// TODO this needs to be implemented in the corresponding cmake file
namespace NeoN
{
#ifdef NeoN_DP_SCALAR
typedef double scalar;
#else
typedef float scalar;
#endif

constexpr scalar ROOTVSMALL = 1e-18;

NEON_INLINE_FUNCTION
scalar mag(const scalar& s) { return std::abs(s); }

// traits for scalar
template<>
NEON_INLINE_FUNCTION scalar one<scalar>()
{
    return 1.0;
};

template<>
NEON_INLINE_FUNCTION scalar zero<scalar>()
{
    return 0.0;
};

template<>
NEON_INLINE_FUNCTION scalar inv<scalar>(scalar in)
{
    return 1.0 / in;
};

} // namespace NeoN
