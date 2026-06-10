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


/**
 * @brief Abstract base for DSL expression optimizers.
 *
 * Subclasses inspect an expression and return a semantically equivalent but
 * potentially more efficient one (e.g. by replacing two operators with a
 * single fused kernel).
 *
 * @tparam ExpressionType  The DSL expression type being optimized.
 */
template<typename ExpressionType>
class Optimizer
{
public:

    virtual ~Optimizer() = default;

    /**
     * @brief Return an optimized copy of @p expr.
     *
     * Implementations must not mutate @p expr; they return a new expression.
     */
    virtual ExpressionType optimize(const ExpressionType& expr) const = 0;
};


/**
 * @brief Fuses separate implicit divergence and Laplacian operators into a
 *        single `GaussGreenDivLaplacian` kernel.
 *
 * When an expression contains both an implicit `DivOperator` and an implicit
 * `LaplacianOperator`, this optimizer replaces them with the combined
 * `GaussGreenDivLaplacian` operator.  The fused operator performs a single
 * face loop instead of two, reducing memory traffic.  If either operator is
 * absent the expression is returned unchanged.
 *
 * @tparam ExpressionType  The DSL expression type being optimized.
 */
template<typename ExpressionType>
class DivLapOptimizer : public Optimizer<ExpressionType>
{
public:

    ~DivLapOptimizer() {}

    ExpressionType optimize(const ExpressionType& expr) const override
    {
        ExpressionType out(expr);

        // early return if not both operators are present
        if (!(expr.template hasOperator<Operator::Type::Implicit>("DivOperator")
              && expr.template hasOperator<Operator::Type::Implicit>("LaplacianOperator")))
        {
            return out;
        }

        using ValueType = typename ExpressionType::ExpressionValueType;
        auto divOperator =
            out.template getOperator<SpatialOperator<ValueType>, Operator::Type::Implicit>(
                   "DivOperator"
            )
                .getConfig();
        auto lapOperator =
            out.template getOperator<SpatialOperator<ValueType>, Operator::Type::Implicit>(
                   "LaplacianOperator"
            )
                .getConfig();

        dsl::SpatialOperator<ValueType> divLapOperator =
            finiteVolume::cellCentred::GaussGreenDivLaplacian<ValueType>(
                expr.exec(), divOperator, lapOperator
            );
        out.addOperator(divLapOperator);

        out.template dropOperator<Operator::Type::Implicit>("DivOperator");
        out.template dropOperator<Operator::Type::Implicit>("LaplacianOperator");

        return out;
    }
};


/**
 * @brief Apply the default optimizer pipeline to an expression.
 *
 * Runs every built-in optimizer (currently `DivLapOptimizer`) in sequence and
 * returns the resulting expression.  Each optimizer may replace, remove, or
 * add operators; later optimizers see the output of earlier ones.
 *
 * @tparam ExpressionType  The DSL expression type.
 * @param  in              Expression to optimize.
 * @return                 Optimized expression.
 */
template<typename ExpressionType>
ExpressionType optimize(const ExpressionType& in)
{
    auto optimizer = std::vector<std::shared_ptr<Optimizer<ExpressionType>>> {
        std::make_shared<DivLapOptimizer<ExpressionType>>()
    };

    return optimize(in, optimizer);
}

/**
 * @brief Apply a caller-supplied optimizer pipeline to an expression.
 *
 * Runs each optimizer in @p opts in order, passing the output of one as the
 * input to the next.
 *
 * @tparam ExpressionType  The DSL expression type.
 * @param  in              Expression to optimize.
 * @param  opts            Ordered list of optimizers to apply.
 * @return                 Optimized expression.
 */
template<typename ExpressionType>
ExpressionType optimize(
    const ExpressionType& in,
    const std::vector<std::shared_ptr<Optimizer<ExpressionType>>>& opts
)
{
    ExpressionType out(in);

    for (const auto& opt : opts)
    {
        out = opt->optimize(out);
    }

    return out;
}


} // namespace dsl
