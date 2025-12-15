// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"

namespace NeoN
{

/**
 * @class SupportsCopyTo
 * @brief MixinClass signaling copyTo is supported
 *
 *  Requires copyToExecutor to be implemented, provides a copyToHost for free
 */
template<typename HostType>
class SupportsCopyTo
{

    // virtual ~SupportsCopyTo() = default;

public:

    [[nodiscard]] virtual HostType copyToExecutor(Executor exec) const = 0;

    [[nodiscard]] HostType copyToHost() const { return copyToExecutor(SerialExecutor()); }
};

}
