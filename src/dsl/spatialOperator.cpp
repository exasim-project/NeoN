// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/dsl/spatialOperator.hpp"

namespace NeoN::dsl
{

// instantiate the template class
template class SpatialOperator<scalar>;
template class SpatialOperator<Vec3>;

} // namespace NeoN::dsl
