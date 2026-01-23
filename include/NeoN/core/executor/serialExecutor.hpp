// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"

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
public:

    using exec = Kokkos::Serial;

    SerialExecutor();
    ~SerialExecutor();

    template<typename T>
    T* alloc(size_t size) const
    {
        // Use standard malloc instead of Kokkos for testing
        return static_cast<T*>(std::malloc(size * sizeof(T)));
    }

    template<typename T>
    T* realloc(void* ptr, size_t newSize) const
    {
        // Use standard realloc instead of Kokkos for testing
        return static_cast<T*>(std::realloc(ptr, newSize * sizeof(T)));
    }

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

    void* alloc(size_t size) const { return std::malloc(size); }

    void* realloc(void* ptr, size_t newSize) const
    {
        return std::realloc(ptr, newSize);
    }

    void free(void* ptr) const noexcept { std::free(ptr); };

    std::string name() const { return "SerialExecutor"; };

    exec underlyingExec() const { return exec {}; }
};

} // namespace NeoN
