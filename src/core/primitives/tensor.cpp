// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/tensor.hpp"

namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << "(" << t[0] << " " << t[1] << " " << t[2] << " " << t[3] << " " << t[4] << " " << t[5]
       << " " << t[6] << " " << t[7] << " " << t[8] << ")";
    return os;
}

} // namespace NeoN
