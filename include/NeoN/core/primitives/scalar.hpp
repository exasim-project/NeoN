// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep
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

KOKKOS_INLINE_FUNCTION
scalar mag(const scalar& s) { return std::abs(s); }

// traits for scalar
template<>
KOKKOS_INLINE_FUNCTION scalar one<scalar>()
{
    return 1.0;
};

template<>
KOKKOS_INLINE_FUNCTION scalar zero<scalar>()
{
    return 0.0;
};

} // namespace NeoN
