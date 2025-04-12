// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <variant>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/tokenList.hpp"

namespace NeoN
{

using Input = std::variant<Dictionary, TokenList>;

template<class DataClass>
DataClass read(Input input)
{
    return std::visit([](const auto& i) { return DataClass::read(i); }, input);
}

}
