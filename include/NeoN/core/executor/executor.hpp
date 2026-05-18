// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <string>
#include <variant>

#include "NeoN/core/executor/serialExecutor.hpp"
#include "NeoN/core/executor/GPUExecutor.hpp"
#include "NeoN/core/executor/CPUExecutor.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/logging.hpp"
#include "NeoN/core/memory/kokkos.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace NeoN
{

using Executor = std::variant<SerialExecutor, CPUExecutor, GPUExecutor>;

/* @brief calls Kokkos::fence to wait for GPU kernels to be finished */
inline void fence(const Executor& exec)
{
    if (std::holds_alternative<NeoN::GPUExecutor>(exec))
    {
        Kokkos::fence();
    }
}


/* @brief returns true if the env var NEON_HARD_DEVICE_SYNC is set to a
 * non-empty, non-"0" value. Cached on first call so subsequent invocations
 * are a single load. Allows A/B testing of the hard-sync path on a single
 * HPC build by flipping the env var between runs. */
inline bool hardDeviceSyncEnabled()
{
    static const bool enabled = []
    {
        const char* s = std::getenv("NEON_HARD_DEVICE_SYNC");
        return s != nullptr && s[0] != '\0' && s[0] != '0';
    }();
    return enabled;
}


/* @brief device-wide synchronization at MPI / linear-solver boundaries.
 *
 * SPUMA-style hardened sync: when the env var `NEON_HARD_DEVICE_SYNC=1`
 * is set at process start AND the executor is GPU on a CUDA build, call
 * `cudaDeviceSynchronize()` directly (mirrors SPUMA's `cudaExecutor::_backendFor`
 * post-kernel sync). This synchronises ALL CUDA streams in the primary
 * context — including streams owned by Ginkgo's CudaExecutor and any
 * UCX-CUDA stream on the primary context — rather than just the Kokkos
 * default execution space (which on newer Kokkos may be per-stream).
 *
 * Otherwise alias to `fence(exec)` to preserve existing behaviour on every
 * non-CUDA build path (CPU, Serial, AMD/HIP, SYCL) and on CUDA builds when
 * the env var is unset.
 *
 * The env var is the only gate so that a single HPC build can be A/B
 * tested by toggling the variable between runs. No rebuild required.
 *
 * Use at every MPI / linear-solver boundary site where the post-MPI /
 * post-solve correctness depends on cross-stream visibility, not just
 * Kokkos-stream completion.
 */
inline void deviceSync(const Executor& exec)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (hardDeviceSyncEnabled() && std::holds_alternative<NeoN::GPUExecutor>(exec))
    {
        cudaDeviceSynchronize();
        return;
    }
#endif
    fence(exec);
}


/* @brief creates highest available executor */
inline Executor createDefaultExecutor(
    std::unique_ptr<AllocatorStrategy> strategy = std::make_unique<DefaultAllocator>()
)
{
#if defined(KOKKOS_ENABLE_CUDA)
    return GPUExecutor {std::move(strategy)};
#elif defined(KOKKOS_ENABLE_HIP)
    return GPUExecutor {std::move(strategy)};
#elif defined(KOKKOS_ENABLE_SYCL)
    return GPUExecutor {std::move(strategy)};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    return CPUExecutor {std::move(strategy)};
#elif defined(KOKKOS_ENABLE_THREADS)
    return CPUExecutor {std::move(strategy)};
#endif
    return SerialExecutor {std::move(strategy)};
}

inline std::string executorName(const Executor& exec)
{
    return std::visit(
        []<typename Exec>(const Exec& concExec) { return concExec.name(); }

        ,
        exec
    );
}

inline MemorySpace memorySpace(const Executor& exec)
{
    return std::visit(
        []<typename Exec>(const Exec& concExec) { return concExec.memorySpace(); }, exec
    );
}

/*@brief convenience function to get access to associated logger */
inline std::shared_ptr<const Logging::BaseLogger> getLogger(const Executor& exec)
{
    return std::visit([](auto e) { return e.getLogger(); }, exec);
}

/*@brief convenience function to get access to associated logger */
inline void setLogger(Executor& exec, std::shared_ptr<Logging::BaseLogger> logger)
{
    std::visit([logger](auto& e) { e.setLogger(logger); }, exec);
}


/**
 * @brief Checks if two executors are equal, i.e. they are of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors are equal, false otherwise.
 */
[[nodiscard]] inline bool operator==(const Executor& lhs, const Executor& rhs)
{
    return std::visit(
        []<typename ExecLhs,
           typename ExecRhs>([[maybe_unused]] const ExecLhs&, [[maybe_unused]] const ExecRhs&)
        {
            if constexpr (std::is_same_v<ExecLhs, ExecRhs>)
            {
                return typename ExecLhs::exec() == typename ExecRhs::exec();
            }
            else
            {
                return false;
            }
        },
        lhs,
        rhs
    );
};

/**
 * @brief Checks if two executors are not equal, i.e. they are not of the same
 * type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors not are equal, false otherwise.
 */
[[nodiscard]] inline bool operator!=(const Executor& lhs, const Executor& rhs)
{
    return !(lhs == rhs);
};

} // namespace NeoN
