// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

/* @file kokkosExecutor.hpp
 * @brief Maps NeoN executors onto Kokkos execution and memory spaces (device TUs only).
 *
 * The executor classes themselves are Kokkos-free so that host-only translation units can hold,
 * copy and compare executors. Everything that needs the Kokkos types lives here.
 */

#if defined(NEON_HOST_ONLY_TU)
#error "kokkosExecutor.hpp includes Kokkos and must not be used in a host-only translation unit"
#endif

#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/CPUExecutor.hpp"
#include "NeoN/core/executor/GPUExecutor.hpp"
#include "NeoN/core/executor/serialExecutor.hpp"

namespace NeoN
{

template<typename ExecutorType>
struct KokkosSpaces;

template<>
struct KokkosSpaces<SerialExecutor>
{
    using exec = Kokkos::Serial;
    using memory = Kokkos::HostSpace;
};

template<>
struct KokkosSpaces<CPUExecutor>
{
    using exec = Kokkos::DefaultHostExecutionSpace;
    using memory = Kokkos::HostSpace;
};

template<>
struct KokkosSpaces<GPUExecutor>
{
    using exec = Kokkos::DefaultExecutionSpace;
    using memory = Kokkos::DefaultExecutionSpace;
};

/* @brief The Kokkos execution space a NeoN executor runs its kernels on. */
template<typename ExecutorType>
using kokkosExecSpace = typename KokkosSpaces<std::remove_cvref_t<ExecutorType>>::exec;

/* @brief Default-constructed instance of the executor's Kokkos execution space. */
template<typename ExecutorType>
kokkosExecSpace<ExecutorType> underlyingExec(const ExecutorType&)
{
    return kokkosExecSpace<ExecutorType> {};
}

/* @brief Unmanaged Kokkos view over memory owned by the given executor. */
template<typename ExecutorType, typename ValueType>
auto createKokkosView(const ExecutorType&, ValueType* ptr, size_t size)
{
    using Memory = typename KokkosSpaces<std::remove_cvref_t<ExecutorType>>::memory;
    return Kokkos::View<ValueType*, Memory, Kokkos::MemoryUnmanaged>(ptr, size);
}

} // namespace NeoN
