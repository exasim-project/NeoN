// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/vec3.hpp"

namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    os << "(" << vec[0] << " " << vec[1] << " " << vec[2] << ")";
    return os;
}

} // namespace NeoN
