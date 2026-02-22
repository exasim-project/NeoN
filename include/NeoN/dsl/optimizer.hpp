// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/dsl/expression.hpp"
#include "NeoN/dsl/spatialOperator.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDdtDivLaplacian.hpp"

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

        dsl::SpatialOperator<typename ExpressionType::ExpressionValueType> divLapOperator =
            finiteVolume::cellCentred::GaussGreenDivLaplacian<
                typename ExpressionType::ExpressionValueType>(
                expr.exec(), divOperator, lapOperator
            );
        out.addOperator(divLapOperator);

        out.dropOperator("DivOperator");
        out.dropOperator("LaplacianOperator");

        return out;
    }
};


/**@brief given an expression subsequent optimiziations are applied */
template<typename ExpressionType>
ExpressionType optimize(const ExpressionType in)
{
    auto optimizer = std::vector<std::shared_ptr<Optimizer<ExpressionType>>> {
        std::make_shared<DivLapOptimizer<ExpressionType>>()
    };

    return optimize(in, optimizer);
}

/**@brief given an expression subsequent optimiziations are applied */
template<typename ExpressionType>
ExpressionType
optimize(const ExpressionType in, std::vector<std::shared_ptr<Optimizer<ExpressionType>>> opts)
{
    ExpressionType out = ExpressionType(in);

    for (auto opt : opts)
    {
        out = opt->optimize(out);
    }

    return out;
}


} // namespace dsl
