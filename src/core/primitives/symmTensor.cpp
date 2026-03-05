// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/symmTensor.hpp"

namespace NeoN
{

std::ostream& operator<<(std::ostream& os, const SymmTensor& s)
{
    os << "(" << s[0] << " " << s[1] << " " << s[2] << " " << s[3] << " " << s[4] << " " << s[5]
       << ")";
    return os;
}

} // namespace NeoN
