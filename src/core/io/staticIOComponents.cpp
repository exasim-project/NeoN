// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/core/io/staticIOComponents.hpp"

namespace NeoFOAM::io
{

std::unique_ptr<StaticIOComponents> StaticIOComponents::ioComponents_;

StaticIOComponents* StaticIOComponents::instance()
{
    if (!ioComponents_)
    {
        ioComponents_.reset(new StaticIOComponents());
    }
    return ioComponents_.get();
}

}
