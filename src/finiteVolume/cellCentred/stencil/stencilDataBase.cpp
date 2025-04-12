// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/finiteVolume/cellCentred/stencil/stencilDataBase.hpp"

std::any& NeoN::StencilDataBase::operator[](const std::string& key) { return stencilDB_.at(key); }

const std::any& NeoN::StencilDataBase::operator[](const std::string& key) const
{
    return stencilDB_.at(key);
}

bool NeoN::StencilDataBase::contains(const std::string& key) const
{
    return stencilDB_.contains(key);
}
