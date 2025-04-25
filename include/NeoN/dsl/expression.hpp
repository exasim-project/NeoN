// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include <vector>

#include "NeoN/core/error.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace la = la;

namespace NeoN::dsl
{


template<typename ValueType>
class Expression
{
public:

    Expression(const Executor& exec) : exec_(exec), temporalOperators_(), spatialOperators_() {}

    Expression(const Expression& exp)
        : exec_(exp.exec_), temporalOperators_(exp.temporalOperators_),
          spatialOperators_(exp.spatialOperators_)
    {}

    /* @brief dispatch read call to operator */
    void read(const Dictionary& input)
    {
        for (auto& op : temporalOperators_)
        {
            op.read(input);
        }
        for (auto& op : spatialOperators_)
        {
            op.read(input);
        }
    }

    /* @brief perform all explicit operation and accumulate the result */
    Vector<ValueType> explicitOperation(localIdx nCells) const
    {
        Vector<ValueType> source(exec_, nCells, zero<ValueType>());
        return explicitOperation(source);
    }

    /* @brief perform all explicit operation and accumulate the result */
    Vector<ValueType> explicitOperation(Vector<ValueType>& source) const
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source);
            }
        }
        return source;
    }

    Vector<ValueType> explicitOperation(Vector<ValueType>& source, scalar t, scalar dt) const
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source, t, dt);
            }
        }
        return source;
    }

    /* @brief perform all implicit operation and accumulate the result */
    void assembleSpatialContributions(la::LinearSystem<ValueType, localIdx>& ls)
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls);
            }
        }
    }

    /* @brief perform all implicit operation and accumulate the result */
    void
    assembleTemporalContributions(la::LinearSystem<ValueType, localIdx>& ls, scalar t, scalar dt)
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls, t, dt);
            }
        }
    }

    void
    assembleLinearSystem(la::LinearSystem<ValueType, localIdx>& ls, scalar t = 0.0, scalar dt = 0.0)
    {
        assembleSpatialContributions(ls);
        if (dt > 0.0)
        {
            assembleTemporalContributions(ls, t, dt);
        }
    }

    void addOperator(const SpatialOperator<ValueType>& oper) { spatialOperators_.push_back(oper); }

    void addOperator(const TemporalOperator<ValueType>& oper)
    {
        temporalOperators_.push_back(oper);
    }

    void addExpression(const Expression& equation)
    {
        for (auto& oper : equation.temporalOperators_)
        {
            temporalOperators_.push_back(oper);
        }
        for (auto& oper : equation.spatialOperators_)
        {
            spatialOperators_.push_back(oper);
        }
    }


    /* @brief getter for the total number of terms in the equation */
    localIdx size() const
    {
        return static_cast<localIdx>(temporalOperators_.size() + spatialOperators_.size());
    }

    // getters
    const std::vector<TemporalOperator<ValueType>>& temporalOperators() const
    {
        return temporalOperators_;
    }

    const std::vector<SpatialOperator<ValueType>>& spatialOperators() const
    {
        return spatialOperators_;
    }

    std::vector<TemporalOperator<ValueType>>& temporalOperators() { return temporalOperators_; }

    std::vector<SpatialOperator<ValueType>>& spatialOperators() { return spatialOperators_; }

    const Executor& exec() const { return exec_; }

private:

    const Executor exec_;

    std::vector<TemporalOperator<ValueType>> temporalOperators_;

    std::vector<SpatialOperator<ValueType>> spatialOperators_;
};

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::VectorValueType>
operator+(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::VectorValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(rhs);
    return expr;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType> operator*(scalar scale, const Expression<ValueType>& es)
{
    Expression<ValueType> expr(es.exec());
    for (const auto& oper : es.temporalOperators())
    {
        expr.addOperator(scale * oper);
    }
    for (const auto& oper : es.spatialOperators())
    {
        expr.addOperator(scale * oper);
    }
    return expr;
}


template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(-1.0 * rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::VectorValueType>
operator-(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::VectorValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(Coeff(-1) * rhs);
    return expr;
}


} // namespace dsl
