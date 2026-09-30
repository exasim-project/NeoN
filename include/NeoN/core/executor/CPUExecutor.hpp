// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"
#include "NeoN/core/memory/allocator.hpp"

namespace NeoN
{

/**
 * @class CPUExecutor
 * @brief Executor for handling multicore CPU based parallelization.
 *
 *
 * @ingroup Executor
 */
class CPUExecutor : public Logging::SupportsLoggingMixin
{
    std::shared_ptr<AllocatorContext> allocContext_ = nullptr;

public:

    CPUExecutor();

    CPUExecutor(std::unique_ptr<AllocatorStrategy> strategy);

    ~CPUExecutor();

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

    void free(void* ptr) const noexcept { allocContext_->free(ptr); }

    MemorySpace memorySpace() const noexcept { return MemorySpace::CPU; }

    std::string name() const { return "CPUExecutor"; };
};

} // namespace NeoN
