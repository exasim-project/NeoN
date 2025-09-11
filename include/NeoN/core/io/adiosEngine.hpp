// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include <memory>

// forward declaration
namespace adios2
{
class ADIOS;
class IO;
class Engine;
}

namespace NeoN::io
{

/*
 * @class AdiosEngine
 * @brief Wrapper for the adios2::Engine instance.
 */
struct AdiosEngine
{
    /*
     * @brief AdiosEngine is default constructable (copyable and moveable).
     */
    AdiosEngine() = default;

    /*
     * @brief Constructor passing the rvalue reference to be stored in enginePtr_.
     *
     * @param io The rvalue reference of an adios2:Engine instance.
     */
    AdiosEngine(adios2::Engine&& engine) : enginePtr_ {std::make_shared<adios2::Engine>(engine)} {};

private:

    std::shared_ptr<adios2::Engine> enginePtr_;
};

} // namespace NeoN::io
