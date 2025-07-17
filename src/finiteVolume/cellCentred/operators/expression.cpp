// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/expression.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// instantiate the template class
template class Expression<scalar>;
// template class Expression<Vec3>;

};
