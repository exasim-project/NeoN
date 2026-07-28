// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "NeoN/core/primitives/traits.hpp"

/**
 * @brief Integer types used throughout NeoN.
 *
 * The following type aliases distinguish between different kinds of integer
 * values:
 *
 * - `label` identifies mesh entities (e.g. cell number and face number)
 * - `localIdx` indexes data that is local to a process or execution space,
 *   such as arrays, views, and local graph structures.
 * - `globalIdx` identifies globally unique entities in distributed-memory
 *   computations.
 * - `size_t` represents sizes and counts (e.g. container sizes, iteration
 *   counts, and memory sizes) and should not be used to identify mesh
 *   entities.
 *
 * The widths of `label`, `localIdx`, and `globalIdx` depend on the compile-time
 * configuration (`NeoN_DP_LABEL` and `NeoN_US_IDX`).
 */

namespace NeoN
{
#ifdef NeoN_DP_LABEL
using label = int64_t;

#ifdef NeoN_US_IDX
using localIdx = uint32_t;
using globalIdx = uint64_t;
#else
using localIdx = int64_t;
using globalIdx = int64_t;
#endif

#else
using label = int32_t;

#ifdef NeoN_US_IDX
using localIdx = uint32_t;
using globalIdx = uint64_t;
#else
using localIdx = int32_t;
using globalIdx = int64_t;
#endif

#endif

using size_t = std::size_t;
using mpi_label_t = int;

// traits for label
template<>
KOKKOS_INLINE_FUNCTION localIdx one<localIdx>()
{
    return 1;
};

template<>
KOKKOS_INLINE_FUNCTION localIdx zero<localIdx>()
{
    return 0;
};

}
