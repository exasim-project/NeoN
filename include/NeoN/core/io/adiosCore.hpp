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

// forward declare
class AdiosConfig;

/*
 * @class AdiosCore
 * @brief Wrapper and storage class for the adios2::ADIOS instance.
 */
class AdiosCore
{
public:

    AdiosCore() { init(); }

    std::shared_ptr<Config> createConfig(const std::string& key) const;

private:

    void init();

    static std::unique_ptr<adios2::ADIOS> adiosPtr_;
};

} // namespace NeoN::io
