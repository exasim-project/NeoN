// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <vector>

#include "NeoN/core/error.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::dsl
{

template<typename VectorType, typename IndexType>
struct PostAssemblyBase
{
    virtual ~PostAssemblyBase() = default;
    virtual void operator()(la::LinearSystem<VectorType, la::CSRMatrix<VectorType, IndexType>>&) {};
};


template<typename ValueType, typename IndexType = localIdx>
class Expression
{
public:

    Expression(const Executor& exec) : exec_(exec), temporalOperators_(), spatialOperators_() {}

    Expression(const Expression& exp)
        : exec_(exp.exec_), temporalOperators_(exp.temporalOperators_),
          spatialOperators_(exp.spatialOperators_)
    {}

    Expression(const SpatialOperator<ValueType>& oper)
        : exec_(oper.exec()), temporalOperators_(), spatialOperators_()
    {
        spatialOperators_.push_back(oper);
    }

    Expression(const TemporalOperator<ValueType>& oper)
        : exec_(oper.exec()), temporalOperators_(), spatialOperators_()
    {
        temporalOperators_.push_back(oper);
    }

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

    /*@brief compute matrix coefficients based on all spatial operators */
    void assembleSpatialOperator(la::LinearSystem<ValueType>& ls) const
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls);
            }
        }
    }

    /*@brief compute matrix coefficients based on all temporal operators
     * assemble directly into linear system
     */
    void assembleTemporalOperator(la::LinearSystem<ValueType>& ls, scalar t, scalar dt) const
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls, t, dt);
            }
        }
    }

    /** @brief construct a linear system and force assembly
     *
     * @param ps a vector of functor performing transformation on the created linear system
     * @return the assembled linear system
     */
    la::LinearSystem<ValueType> assemble(
        const UnstructuredMesh& mesh,
        scalar t,
        scalar dt,
        std::span<const PostAssemblyBase<ValueType, IndexType>> ps = {}
    ) const
    {
        auto ls = la::createEmptyLinearSystem<ValueType>(mesh);
        assemble(t, dt, ls, ps);
        return ls;
    }

    /* @brief assemble into a given linear system
     *
     * @param ps a vector of functor performing transformation on the created linear system
     */
    void assemble(
        scalar t,
        scalar dt,
        la::LinearSystem<ValueType>& ls,
        std::span<const PostAssemblyBase<ValueType, IndexType>> ps = {}
    ) const
    {
        assembleSpatialOperator(ls);         // add spatial operator
        assembleTemporalOperator(ls, t, dt); // add temporal operators

        // perform post assembly transformations
        for (auto p : ps)
        {
            p(ls);
        }
    }

    void addOperator(const SpatialOperator<ValueType>& oper) { spatialOperators_.push_back(oper); }

    void addOperator(const TemporalOperator<ValueType>& oper)
    {
        temporalOperators_.push_back(oper);
    }

    void addExpression(const Expression& equation)
    {
        for (auto& op : equation.temporalOperators_)
        {
            temporalOperators_.push_back(op);
        }
        for (auto& op : equation.spatialOperators_)
        {
            spatialOperators_.push_back(op);
        }
    }

    /**@brief returns operator of given type and name exists */
    template<typename OperatorType, Operator::Type Type>
    bool hasOperatorOfType(const std::string& name) const
    {
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if constexpr (std::is_same_v<OperatorType, SpatialOperator<ValueType>>)
        {
            return std::ranges::any_of(spatialOperators_, matchNameAndType);
        }
        else if constexpr (std::is_same_v<OperatorType, TemporalOperator<ValueType>>)
        {
            return std::ranges::any_of(temporalOperators_, matchNameAndType);
        }
        return false;
    }

    /**@brief returns whether the expression contains an operator with a given name */
    template<Operator::Type Type>
    bool hasOperator(const std::string& name) const
    {
        return hasOperatorOfType<SpatialOperator<ValueType>, Type>(name)
            || hasOperatorOfType<TemporalOperator<ValueType>, Type>(name);
    }

    /**@brief returns operator of given type and name */
    template<typename OperatorType, Operator::Type Type>
    OperatorType& getOperator(const std::string& name)
    {
        if (!hasOperatorOfType<OperatorType, Type>(name))
        {
            throw std::runtime_error {"No operator with given name and type found"};
        }
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if constexpr (std::is_same_v<OperatorType, SpatialOperator<ValueType>>)
        {
            return *std::ranges::find_if(spatialOperators_, matchNameAndType);
        }
        else if constexpr (std::is_same_v<OperatorType, TemporalOperator<ValueType>>)
        {
            return *std::ranges::find_if(temporalOperators_, matchNameAndType);
        }
        throw std::runtime_error {"Unknown operator type"};
        // should never be reached, shut up compiler warning
        return spatialOperators_[0];
    }

    /**@brief removes operator of given name */
    template<Operator::Type Type>
    void dropOperator(const std::string& name)
    {
        if (!hasOperator<Type>(name))
        {
            throw std::runtime_error {"No operator with given name and type found"};
        }
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if (hasOperatorOfType<SpatialOperator<ValueType>, Type>(name))
        {
            std::erase_if(spatialOperators_, matchNameAndType);
        }
        else
        {
            std::erase_if(temporalOperators_, matchNameAndType);
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
