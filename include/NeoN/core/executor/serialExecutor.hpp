// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"
#include "NeoN/core/memory/allocator.hpp"

#include <Kokkos_Core.hpp>

namespace NeoN
{

/**
 * @class SerialExecutor
 * @brief Reference executor for serial CPU execution.
 *
 * @ingroup Executor
 */
class SerialExecutor : public Logging::SupportsLoggingMixin
{
    std::shared_ptr<AllocatorContext> allocContext_ = nullptr;

public:

    using exec = Kokkos::Serial;

    SerialExecutor();

    SerialExecutor(std::unique_ptr<AllocatorStrategy> strategy);

    ~SerialExecutor();

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

    MemorySpace memorySpace() const noexcept { return MemorySpace::CPU; }

    /** @brief create a Kokkos view for a given ptr
     *
     * Based on the executor this function creates a Kokkos view into the data managed by ptr
     * @param ptr Pointer to data for which a view should be created
     * @param size Number of elements this view contains
     * @tparam ValueType The value type the underlying memory holds
     * */
    template<typename ValueType>
    decltype(auto) createKokkosView(ValueType* ptr, size_t size) const
    {
        return Kokkos::View<ValueType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(ptr, size);
    }

    std::string name() const { return "SerialExecutor"; };

    exec underlyingExec() const { return exec {}; }
};

} // namespace NeoN
