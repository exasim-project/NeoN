// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <memory>

namespace NeoFOAM::io
{

/*
 * @class Core
 * @brief Type-erased interface to store the IO core like adios2::ADIOS.
 *
 * The Core class is a type-erased interface to store core components from IO
 * libraries wrapped into classes that cohere with the Core's interface.
 */
class Core
{
public:

    /*
     * @brief Constructs a Core component for IO from a specific CoreType.
     *
     * @tparam CoreType The core type that wraps a specific IO library base.
     * @param core The core instance to be wrapped.
     */
    template<typename CoreType>
    Core(CoreType core) : pimpl_(std::make_unique<CoreModel<CoreType>>(std::move(core)))
    {}

private:

    /*
     * @brief Base concept declaring the type-erased interface
     */
    struct CoreConcept
    {
        virtual ~CoreConcept() = default;
    };

    /*
     * @brief Derived model delegating the implementation to the type-erased PIMPL
     */
    template<typename CoreType>
    struct CoreModel : CoreConcept
    {
        CoreModel(CoreType core) : core_(std::move(core)) {}
        CoreType core_;
    };

    /*
     * @brief Type-erased PIMPL
     */
    std::unique_ptr<CoreConcept> pimpl_;
};

} // namespace NeoFOAM::io
