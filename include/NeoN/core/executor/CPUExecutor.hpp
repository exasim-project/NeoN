// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

namespace NeoN
{

/**
 * @class CPUExecutor
 * @brief Executor for handling multicore CPU based parallelization.
 *
 *
 * @ingroup Executor
 */
class CPUExecutor
{
public:

    using exec = Kokkos::DefaultHostExecutionSpace;

    CPUExecutor();
    ~CPUExecutor();

    template<typename T>
    T* alloc(size_t size) const
    {
        return static_cast<T*>(Kokkos::kokkos_malloc<exec>("Field", size * sizeof(T)));
    }

    template<typename T>
    T* realloc(void* ptr, size_t newSize) const
    {
        return static_cast<T*>(Kokkos::kokkos_realloc<exec>(ptr, newSize * sizeof(T)));
    }

    void* alloc(size_t size) const { return Kokkos::kokkos_malloc<exec>("Field", size); }

    void* realloc(void* ptr, size_t newSize) const
    {
        return Kokkos::kokkos_realloc<exec>(ptr, newSize);
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

    void free(void* ptr) const noexcept { Kokkos::kokkos_free<exec>(ptr); };

    std::string name() const { return "CPUExecutor"; };

    exec underlyingExec() const { return exec {}; }
};

} // namespace NeoN
