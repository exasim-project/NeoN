// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#include "NeoN/core/io/staticIOComponents.hpp"

// FIXME This is only for compile tests at the moment.
#include "NeoN/core/io/adiosCore.hpp"

namespace NeoN::io
{

std::unique_ptr<StaticIOComponents> StaticIOComponents::ioComponents_;

StaticIOComponents* StaticIOComponents::instance()
{
    if (!ioComponents_)
    {
        ioComponents_.reset(new StaticIOComponents());

        // FIXME This is only for compile tests at the moment.
        ioComponents_->core_ = std::make_unique<Core>(new Core(new AdiosCore()));
    }
    return ioComponents_.get();
}

}
