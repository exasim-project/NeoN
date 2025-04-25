// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/primitives/vec3.hpp"

namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    os << "(" << vec[0] << " " << vec[1] << " " << vec[2] << ")";
    return os;
}

} // namespace NeoN
