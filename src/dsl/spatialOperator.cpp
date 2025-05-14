// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 NeoN authors

#include "NeoN/dsl/spatialOperator.hpp"

namespace NeoN::dsl
{

// instantiate the template class
template class SpatialOperator<scalar>;
template class SpatialOperator<Vec3>;

} // namespace NeoN::dsl
