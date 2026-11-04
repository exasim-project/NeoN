// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/symmTensor.hpp"


namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const SymmTensor& s)
{
    os << "((" << s(0, 0) << " " << s(0, 1) << " " << s(0, 2) << ") (" << s(1, 0) << " " << s(1, 1)
       << " " << s(1, 2) << ") (" << s(2, 0) << " " << s(2, 1) << " " << s(2, 2) << "))";
    return os;
}

} // namespace NeoN
