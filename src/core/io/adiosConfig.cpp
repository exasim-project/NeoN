// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include <adios2.h>

#include "NeoFOAM/core/io/staticIOComponents.hpp"
#include "NeoFOAM/core/io/adiosCore.hpp"
#include "NeoFOAM/core/io/adiosConfig.hpp"
#include "NeoFOAM/core/io/config.hpp"

namespace NeoFOAM::io
{

std::shared_ptr<Engine> AdiosConfig::createEngine(const std::string& path) const
{
    StaticIOComponents* components = StaticIOComponents::instance();
    auto key = configPtr_->Name() + path;
    std::shared_ptr<Engine> adiosEngine;
    adiosEngine = components->at(key, adiosEngine);
    if (!adiosEngine)
    {
        // TODO Where to BeginStep?
        adiosEngine =
            std::make_shared<Engine>(new Engine(configPtr_->Open(path, adios2::Mode::Append)));
        components->insert(key, adiosEngine);
    }
    return adiosEngine;
}

} // namespace NeoFOAM
