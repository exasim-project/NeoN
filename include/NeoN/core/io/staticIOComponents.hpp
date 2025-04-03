// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>

#include "core.hpp"
#include "config.hpp"
#include "engine.hpp"

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

    using core_component = Core;
    using config_map = std::unordered_map<std::string, std::shared_ptr<Config>>;
    using engine_map = std::unordered_map<std::string, std::shared_ptr<Engine>>;

    std::unique_ptr<core_component> core_;
    std::unique_ptr<config_map> configMap_;
    std::unique_ptr<engine_map> engineMap_;

    /**
     * @brief Private constructor of the singleton instance ioComponents_
     */
    StaticIOComponents();

public:

    /**
     * @brief Retrieves the singleton instance of StaticIOComponents
     */
    static StaticIOComponents* instance();

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
