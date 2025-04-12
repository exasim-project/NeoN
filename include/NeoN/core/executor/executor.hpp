// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <string>
#include <variant>

#include "NeoN/core/executor/serialExecutor.hpp"
#include "NeoN/core/executor/GPUExecutor.hpp"
#include "NeoN/core/executor/CPUExecutor.hpp"
#include "NeoN/core/error.hpp"

namespace NeoN
{

using Executor = std::variant<SerialExecutor, CPUExecutor, GPUExecutor>;

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
