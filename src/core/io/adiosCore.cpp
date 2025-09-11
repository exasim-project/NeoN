// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#include <adios2.h>

#include "NeoN/core/io/staticIOComponents.hpp"
#include "NeoN/core/io/adiosCore.hpp"
#include "NeoN/core/io/adiosConfig.hpp"
#include "NeoN/core/io/config.hpp"

namespace NeoN::io
{

std::unique_ptr<adios2::ADIOS> AdiosCore::adiosPtr_;

std::shared_ptr<Config> AdiosCore::createConfig(const std::string& name) const
{
    StaticIOComponents* components = StaticIOComponents::instance();
    std::shared_ptr<Config> adiosConfig;

    adiosConfig = components->at(name, adiosConfig);

    if (!adiosConfig)
    {
        adiosConfig = std::make_shared<Config>(new Config(AdiosConfig(adiosPtr_->DeclareIO(name))));
        components->insert(name, adiosConfig);
    }

    return adiosConfig;
}

void AdiosCore::init()
{
    if (!adiosPtr_)
    {
        // Constructor for non-MPI (serial) application
        adiosPtr_.reset(new adios2::ADIOS());
        // TODO Add construction for MPI application
        //      once NeoN supports MPI runs.
        // TODO Add configuration file.
    }
}

} // namespace NeoN
