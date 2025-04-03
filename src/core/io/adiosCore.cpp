// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include <adios2.h>

#include "NeoFOAM/core/io/staticIOComponents.hpp"
#include "NeoFOAM/core/io/adiosCore.hpp"
#include "NeoFOAM/core/io/adiosConfig.hpp"
#include "NeoFOAM/core/io/config.hpp"

namespace NeoFOAM::io
{

std::unique_ptr<adios2::ADIOS> AdiosCore::adiosPtr_;

std::shared_ptr<Config> AdiosCore::createConfig()
{
    StaticIOComponents* components = StaticIOComponents::instance();
    std::shared_ptr<Config> adiosConfig;

    // TODO implement AdiosConfig, its construction and declaration
    // std::shared_ptr<AdiosConfig> adiosConfig;

    return adiosConfig;
}

void AdiosCore::init()
{
    if (!adiosPtr_)
    {
        // Constructor for non-MPI (serial) application
        adiosPtr_.reset(new adios2::ADIOS());
        // TODO Add construction for MPI application
        //      once NeoFOAM supports MPI runs.
        // TODO Add configuration file.
    }
}

} // namespace NeoFOAM
