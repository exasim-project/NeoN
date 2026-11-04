// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/tensor.hpp"


namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    os << "((" << t(0, 0) << " " << t(0, 1) << " " << t(0, 2) << ") (" << t(1, 0) << " " << t(1, 1)
       << " " << t(1, 2) << ") (" << t(2, 0) << " " << t(2, 1) << " " << t(2, 2) << "))";
    return os;
}

} // namespace NeoN
