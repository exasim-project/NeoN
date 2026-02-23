// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockDsl.hpp"

namespace NeoN::bdsl::imp
{

SpatialOperator<scalar>
source(scalar coeff, const Vector<scalar>& /*field*/, const std::string& fieldName)
{
    return SpatialOperator<scalar>(BlockSourceTerm(coeff, fieldName));
}

} // namespace NeoN::bdsl::imp
