// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>

#include "NeoN/core/executor/serialExecutor.hpp"
#include "NeoN/core/executor/GPUExecutor.hpp"
#include "NeoN/core/executor/CPUExecutor.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/logging.hpp"
#include "NeoN/core/memory/kokkos.hpp"

namespace NeoN
{

using Executor = std::variant<SerialExecutor, CPUExecutor, GPUExecutor>;

/* @brief calls Kokkos::fence to wait for GPU kernels to be finished */
void fence(const Executor& exec);


/* @brief creates highest available executor */
Executor createDefaultExecutor(
    std::unique_ptr<AllocatorStrategy> strategy = std::make_unique<DefaultAllocator>()
);

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
                // Executors of the same type run on the default instance of the same Kokkos
                // execution space, which always compare equal.
                return true;
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
