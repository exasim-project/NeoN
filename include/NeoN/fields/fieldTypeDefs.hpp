// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"


namespace NeoFOAM
{

using labelField = NeoFOAM::Field<label>;
using scalarField = NeoFOAM::Field<scalar>;
using vectorField = NeoFOAM::Field<Vector>;

} // namespace NeoFOAM
