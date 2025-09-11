// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#include <adios2.h>

#include "NeoN/core/io/staticIOComponents.hpp"
#include "NeoN/core/io/adiosCore.hpp"
#include "NeoN/core/io/adiosConfig.hpp"
#include "NeoN/core/io/config.hpp"

namespace NeoN::io
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

} // namespace NeoN
