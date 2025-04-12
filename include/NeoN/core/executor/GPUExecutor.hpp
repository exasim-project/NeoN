// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>

namespace NeoFOAM
{

/**
 * @class GPUExecutor
 * @brief Executor for GPU offloading.
 *
 * @ingroup Executor
 */
class GPUExecutor
{
public:

    using exec = Kokkos::DefaultExecutionSpace;

    GPUExecutor();
    ~GPUExecutor();

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
        return Kokkos::View<ValueType*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryUnmanaged>(
            ptr, size
        );
    }

    void* alloc(size_t size) const { return Kokkos::kokkos_malloc<exec>("Field", size); }

    void* realloc(void* ptr, size_t newSize) const
    {
        return Kokkos::kokkos_realloc<exec>(ptr, newSize);
    }

    void free(void* ptr) const noexcept { Kokkos::kokkos_free<exec>(ptr); }

    std::string name() const { return "GPUExecutor"; };

    exec underlyingExec() const { return exec {}; }
};

} // namespace NeoFOAM
