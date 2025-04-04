// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <memory>

// forward declaration
namespace adios2
{
class ADIOS;
class IO;
class Engine;
}

namespace NeoFOAM::io
{

/*
 * @class AdiosConfig
 * @brief Wrapper for the adios2::IO instance.
 */
struct AdiosConfig
{
    /*
     * @brief AdiosConfig is default constructable (copyable and moveable).
     */
    AdiosConfig() = default;

    /*
     * @brief Constructor passing the rvalue reference to be stored in configPtr_.
     *
     * @param io The rvalue reference of an adios2:IO instance.
     */
    AdiosConfig(adios2::IO&& io) : configPtr_ {std::make_shared<adios2::IO>(io)} {};

private:

    std::shared_ptr<adios2::IO> configPtr_;
};

} // namespace NeoFOAM::io
