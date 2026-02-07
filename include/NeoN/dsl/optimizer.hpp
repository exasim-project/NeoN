// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// #include <vector>

// #include "NeoN/core/error.hpp"
// #include "NeoN/core/primitives/scalar.hpp"
// #include "NeoN/fields/field.hpp"
// #include "NeoN/linearAlgebra/linearSystem.hpp"
// #include "NeoN/dsl/spatialOperator.hpp"
// #include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/dsl/expression.hpp"

// #include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
// #include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::dsl
{


template<typename ValueType>
class Optimizer
{
};


/**@brief given an */
template<typename ExpressionType>
ExpressionType optimize(const ExpressionType in)
{

    auto spatialOperators = in.spatialOperators();

    for (auto op : spatialOperators)
    {
        std::cout << __FILE__ << ":" << op.getName() << "\n";
    }

    return in;
}


} // namespace dsl
