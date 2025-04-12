// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/primitives/vector.hpp"

namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const Vector& vec)
{
    os << "(" << vec[0] << " " << vec[1] << " " << vec[2] << ")";
    return os;
}

} // namespace NeoN
