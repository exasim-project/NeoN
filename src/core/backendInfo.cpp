// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/backendInfo.hpp"

#include <Kokkos_Core.hpp>

namespace NeoN
{

bool hasSerialBackend() noexcept { return true; }

bool hasCpuBackend() noexcept
{
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    return true;
#else
    return false;
#endif
}

bool hasGpuBackend() noexcept
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    constexpr bool deviceCannotAccessHost =
        !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
    return deviceCannotAccessHost;
#else
    return false;
#endif
}

} // namespace NeoN
