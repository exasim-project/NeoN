// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>

namespace NeoFOAM::io
{

/**
 * @class StaticIOComponents
 * @brief A collection of IO components that must be initialized only once during program life time.
 *
 * The StaticIOComponents class renders a database for IO components that must not be
 * recreated/reinitialized during program life time like adios2::ADIOS. Plus the class allows
 * storing IO resources for asynchronous IO.
 */
class StaticIOComponents
{
    static std::unique_ptr<StaticIOComponents> ioComponents_;

    /**
     * @brief Private constructor of the singleton instance ioComponents_
     */
    StaticIOComponents();

public:

    /**
     * @brief Retrieves the singleton instance of StaticIOComponents
     */
    StaticIOComponents* instance();

    /**
     * @brief Deletes the singleton instance of StaticIOComponents
     */
    void deleteInstance();

    /**
     * @brief StaticIOComponent is not copyable
     */
    StaticIOComponents(StaticIOComponents&) = delete;

    /**
     * @brief StaticIOComponent is not copyable
     */
    StaticIOComponents& operator=(const StaticIOComponents&) = delete;

    /*
     * @brief Destructor to finalize all remaining components
     */
    ~StaticIOComponents() = default;
};

} // namespace NeoFOAM::io
