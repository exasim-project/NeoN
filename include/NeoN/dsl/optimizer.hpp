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
#include "NeoN/dsl/spatialOperator.hpp"

// #include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"

namespace NeoN::dsl
{


template<typename ExpressionType>
class Optimizer
{
public:

    virtual ~Optimizer() = default;
    virtual ExpressionType optimize(const ExpressionType& expr) const = 0;
};

template<typename ExpressionType>
class DivLapOptimizer : public Optimizer<ExpressionType>
{
public:

    ~DivLapOptimizer() {}

    ExpressionType optimize(const ExpressionType& expr) const override
    {
        ExpressionType out(expr);

        // early return if not both operators are present
        if (!(expr.hasOperator("DivOperator") && expr.hasOperator("LaplacianOperator")))
        {
            return out;
        }

        auto divOperator =
            out.template getOperator<SpatialOperator<typename ExpressionType::ExpressionValueType>>(
                   "DivOperator"
            )
                .getConfig();
        auto lapOperator =
            out.template getOperator<SpatialOperator<typename ExpressionType::ExpressionValueType>>(
                   "LaplacianOperator"
            )
                .getConfig();

        out.dropOperator("DivOperator");
        out.dropOperator("LaplacianOperator");

        dsl::SpatialOperator<typename ExpressionType::ExpressionValueType> divLapOperator =
            finiteVolume::cellCentred::GaussGreenDivLaplacian<
                typename ExpressionType::ExpressionValueType>(
                expr.exec(), divOperator, lapOperator
            );
        out.addOperator(divLapOperator);

        return out;
    }
};


/**@brief given an expression subsequent optimiziations are applied */
template<typename ExpressionType>
ExpressionType optimize(const ExpressionType in)
{
    ExpressionType out = ExpressionType(in);
    auto optimizer = std::vector<std::shared_ptr<Optimizer<ExpressionType>>> {
        std::make_shared<DivLapOptimizer<ExpressionType>>()
    };

    for (auto opt : optimizer)
    {
        out = opt->optimize(out);
    }

    return out;
}


} // namespace dsl
