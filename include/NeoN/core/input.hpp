// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <variant>

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/tokenList.hpp"

namespace NeoFOAM
{

using Input = std::variant<Dictionary, TokenList>;

template<class DataClass>
DataClass read(Input input)
{
    return std::visit([](const auto& i) { return DataClass::read(i); }, input);
}

}
