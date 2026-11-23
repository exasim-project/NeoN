// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include <Kokkos_Core.hpp>

#include "NeoN/core/memory/allocator.hpp"
#include "NeoN/core/primitives/label.hpp"

namespace NeoN
{

/**@class KokkosAllocactor
 * @brief allocate and free memory via kokkos
 */
class KokkosAllocator : public AllocatorStrategy
{

public:

    using GPUMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using CPUMemSpace = Kokkos::HostSpace;

    void* alloc(sizeType size) override;

    void* realloc(void* ptr, sizeType size) override;

    void free(void* ptr) override;

    ~KokkosAllocator() override {}
};

using DefaultAllocator = KokkosAllocator;

}
