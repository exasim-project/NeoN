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
    AdiosConfig() = default;
    AdiosConfig(adios2::IO& io) : configPtr_ {std::make_unique<adios2::IO>(io)} {};

    std::unique_ptr<adios2::IO> configPtr_;
};

} // namespace NeoFOAM::io
