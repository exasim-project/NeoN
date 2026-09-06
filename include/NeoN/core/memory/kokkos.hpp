// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include <Kokkos_Core.hpp>

#include "NeoN/core/memory/allocator.hpp"

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

    void* alloc(size_t size) override;

    void* realloc(void* ptr, size_t size) override;

    void free(void* ptr) override;

    ~KokkosAllocator() override {}
};

using DefaultAllocator = KokkosAllocator;

}
