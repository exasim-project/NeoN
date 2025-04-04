// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <type_traits>

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

    template<typename ComponentType>
    auto getMap(ComponentType const* const component) const
    {
        if constexpr (std::is_same<ComponentType, Config>::value)
        {
            return configMap_.get();
        }
        else
        {
            return engineMap_.get();
        }
    }

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

    /*
     * @brief Retrieve an IO component with specific key from the respective map.
     */
    template<typename ComponentType>
    auto at(const std::string& key, std::shared_ptr<ComponentType>& componentPtr)
    {
        auto componentMap = getMap(componentPtr.get());
        if (componentMap->count(key))
        {
            componentPtr = componentMap->at(key);
        }
        return componentPtr;
    }

    /*
     * @brief Add an IO component with specific key to the respective map.
     */
    template<typename ComponentType>
    void insert(const std::string& key, std::shared_ptr<ComponentType>& componentPtr)
    {
        auto componentMap = getMap(componentPtr.get());
        if (!componentMap->count(key))
        {
            componentMap->insert({key, componentPtr});
        }
    }
};

} // namespace NeoFOAM::io
