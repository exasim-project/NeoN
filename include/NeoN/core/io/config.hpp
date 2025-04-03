// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <memory>

namespace NeoFOAM::io
{

/*
 * @class Config
 * @brief Type-erased interface to store the IO configuration like adios2::IO.
 *
 * The Config class is a type-erased interface to store configurational components
 * from IO libraries wrapped into classes that cohere with the Config's interface.
 */
class Config
{
public:

    /*
     * @brief Constructs a Config component for IO from a specific ConfigType.
     *
     * @tparam ConfigType The configuration type that wraps a specific IO library configuration.
     * @param config The configuration instance to be wrapped.
     */
    template<typename ConfigType>
    Config(ConfigType config) : pimpl_(std::make_unique<ConfigModel<ConfigType>>(std::move(config)))
    {}

private:

    /*
     * @brief Base concept declaring the type-erased interface
     */
    struct ConfigConcept
    {
        virtual ~ConfigConcept() = default;
    };

    /*
     * @brief Derived model delegating the implementation to the type-erased PIMPL
     */
    template<typename ConfigType>
    struct ConfigModel : ConfigConcept
    {
        ConfigModel(ConfigType config) : config_(std::move(config)) {}
        ConfigType config_;
    };

    /*
     * @brief Type-erased PIMPL
     */
    std::unique_ptr<ConfigConcept> pimpl_;
};

} // namespace NeoFOAM::io
