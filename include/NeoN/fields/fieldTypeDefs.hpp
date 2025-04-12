// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"


namespace NeoN
{

using labelField = NeoN::Field<label>;
using scalarField = NeoN::Field<scalar>;
using vectorField = NeoN::Field<Vec3>;

} // namespace NeoN
