// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <memory>

namespace NeoFOAM::io
{

/*
 * @class Engine
 * @brief Type-erased interface to store the IO engine/stream like adios2::Engine.
 *
 * The Engine class is a type-erased interface to store engine/streaming components from IO
 * libraries wrapped into classes that cohere with the Engine's interface.
 */
class Engine
{
public:

    /*
     * @brief Constructs an Engine component for IO from a specific EngineType.
     *
     * @tparam EngineType The engine type that wraps a specific IO library engine/stream.
     * @param engine The engine instance to be wrapped.
     */
    template<typename EngineType>
    Engine(EngineType core) : pimpl_(std::make_unique<EngineModel<EngineType>>(std::move(core)))
    {}

private:

    /*
     * @brief Base concept declaring the type-erased interface
     */
    struct EngineConcept
    {
        virtual ~EngineConcept() = default;
    };

    /*
     * @brief Derived model delegating the implementation to the type-erased PIMPL
     */
    template<typename EngineType>
    struct EngineModel : EngineConcept
    {
        EngineModel(EngineType core) : core_(std::move(core)) {}
        EngineType core_;
    };

    /*
     * @brief Type-erased PIMPL
     */
    std::unique_ptr<EngineConcept> pimpl_;
};

} // namespace NeoFOAM::io
