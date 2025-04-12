// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vector.hpp"


namespace NeoN
{

using labelField = NeoN::Field<label>;
using scalarField = NeoN::Field<scalar>;
using vectorField = NeoN::Field<Vector>;

} // namespace NeoN
