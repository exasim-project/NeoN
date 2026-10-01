// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/memory/allocator.hpp"

namespace NeoN
{

/**@class KokkosAllocactor
 * @brief allocate and free memory via kokkos
 */
class KokkosAllocator : public AllocatorStrategy
{

public:

    void* alloc(size_t size) override;

    void* realloc(void* ptr, size_t size) override;

    void free(void* ptr) override;

    ~KokkosAllocator() override {}
};

using DefaultAllocator = KokkosAllocator;

}
