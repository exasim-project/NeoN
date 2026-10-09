// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/executor.hpp"

namespace NeoN
{

void fence(const Executor& exec)
{
    if (std::holds_alternative<NeoN::GPUExecutor>(exec))
    {
        Kokkos::fence();
    }
}

Executor createDefaultExecutor(std::unique_ptr<AllocatorStrategy> strategy)
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    return GPUExecutor {std::move(strategy)};
#elif defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    return CPUExecutor {std::move(strategy)};
#else
    return SerialExecutor {std::move(strategy)};
#endif
}

} // namespace NeoN
