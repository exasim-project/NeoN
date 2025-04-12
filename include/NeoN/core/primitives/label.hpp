// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <cstdint>

#include "NeoN/core/primitives/traits.hpp"


namespace NeoN
{
#ifdef NeoN_DP_LABEL
using label = int64_t;
using localIdx = uint64_t;
#else
using label = int32_t;
using localIdx = uint32_t;
#endif
using globalIdx = uint64_t;
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
