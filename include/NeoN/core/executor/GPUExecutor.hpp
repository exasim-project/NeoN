// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"
#include "NeoN/core/memory/allocator.hpp"

namespace NeoN
{

/**
 * @class GPUExecutor
 * @brief Executor for GPU offloading.
 *
 * @ingroup Executor
 */
class GPUExecutor : public Logging::SupportsLoggingMixin
{

    std::shared_ptr<AllocatorContext> allocContext_ = nullptr;

public:

    GPUExecutor();

    GPUExecutor(std::unique_ptr<AllocatorStrategy> strategy);

    ~GPUExecutor();

    /*@brief allocation of size elements of sizeof(T) bytes */
    template<typename T>
    T* alloc(size_t elements) const
    {
        if (!allocContext_)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        return allocContext_->alloc<T>(elements);
    }

    template<typename T>
    T* realloc(void* ptr, size_t elements) const
    {
        if (!allocContext_)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        return allocContext_->realloc<T>(ptr, elements);
    }

    void free(void* ptr) const noexcept
    {
        if (!allocContext_)
        {
            NF_ERROR_EXIT("No allocator set");
        }
        allocContext_->free(ptr);
    }

    MemorySpace memorySpace() const noexcept { return MemorySpace::GPU; }

    std::string name() const { return "GPUExecutor"; };
};

} // namespace NeoN
