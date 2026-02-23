// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/blockMatrixView.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::bdsl
{

/**
 * @class BlockSourceTerm
 * @brief A source term that targets a named field in a block-coupled system.
 *
 * Stores a scalar coefficient and the name of the field it couples to.
 * implicitOperation writes the coefficient into the block matrix diagonal.
 */
class BlockSourceTerm
{

public:

    using VectorValueType = scalar;

    BlockSourceTerm(scalar coefficient, std::string fieldName);

    [[nodiscard]] std::string getFieldName() const;
    [[nodiscard]] std::string getName() const;
    [[nodiscard]] scalar coefficient() const;

    void implicitOperation(
        la::BlockMatrixView bmView,
        la::SparsityView<localIdx> spView,
        localIdx eqI,
        localIdx colJ,
        localIdx nCells,
        const Executor& exec
    ) const;

private:

    scalar coefficient_;
    std::string fieldName_;
};


/**
 * @brief Concept for types that can act as block spatial operators.
 */
template<typename T>
concept IsBlockSpatialOperator = requires(T const t) {
    {
        t.getFieldName()
    } -> std::convertible_to<std::string>;
    {
        t.getName()
    } -> std::convertible_to<std::string>;
    {
        t.implicitOperation(
            std::declval<la::BlockMatrixView>(),
            std::declval<la::SparsityView<localIdx>>(),
            std::declval<localIdx>(),
            std::declval<localIdx>(),
            std::declval<localIdx>(),
            std::declval<const Executor&>()
        )
    } -> std::same_as<void>;
};


/**
 * @class SpatialOperator
 * @brief Type-erased wrapper for block spatial operators.
 *
 * Uses the Concept/Model pattern (like dsl::SpatialOperator) but with
 * getFieldName() and block-aware implicitOperation signature.
 */
template<typename ValueType>
class SpatialOperator
{

public:

    template<IsBlockSpatialOperator T>
    SpatialOperator(T cls) : model_(std::make_unique<OperatorModel<T>>(std::move(cls)))
    {}

    SpatialOperator(const SpatialOperator& other) : model_(other.model_->clone()) {}

    SpatialOperator(SpatialOperator&& other) : model_(std::move(other.model_)) {}

    SpatialOperator& operator=(const SpatialOperator& other)
    {
        model_ = other.model_->clone();
        return *this;
    }

    SpatialOperator& operator=(SpatialOperator&& other)
    {
        model_ = std::move(other.model_);
        return *this;
    }

    [[nodiscard]] std::string getFieldName() const { return model_->getFieldName(); }

    [[nodiscard]] std::string getName() const { return model_->getName(); }

    void implicitOperation(
        la::BlockMatrixView bmView,
        la::SparsityView<localIdx> spView,
        localIdx eqI,
        localIdx colJ,
        localIdx nCells,
        const Executor& exec
    ) const
    {
        model_->implicitOperation(bmView, spView, eqI, colJ, nCells, exec);
    }

private:

    struct OperatorConcept
    {
        virtual ~OperatorConcept() = default;
        virtual std::string getFieldName() const = 0;
        virtual std::string getName() const = 0;
        virtual void implicitOperation(
            la::BlockMatrixView bmView,
            la::SparsityView<localIdx> spView,
            localIdx eqI,
            localIdx colJ,
            localIdx nCells,
            const Executor& exec
        ) const = 0;
        virtual std::unique_ptr<OperatorConcept> clone() const = 0;
    };

    template<typename ConcreteType>
    struct OperatorModel : OperatorConcept
    {
        OperatorModel(ConcreteType op) : op_(std::move(op)) {}

        std::string getFieldName() const override { return op_.getFieldName(); }

        std::string getName() const override { return op_.getName(); }

        void implicitOperation(
            la::BlockMatrixView bmView,
            la::SparsityView<localIdx> spView,
            localIdx eqI,
            localIdx colJ,
            localIdx nCells,
            const Executor& exec
        ) const override
        {
            op_.implicitOperation(bmView, spView, eqI, colJ, nCells, exec);
        }

        std::unique_ptr<OperatorConcept> clone() const override
        {
            return std::make_unique<OperatorModel>(*this);
        }

        ConcreteType op_;
    };

    std::unique_ptr<OperatorConcept> model_;
};


/**
 * @brief Combine two spatial operators into a vector.
 */
template<typename ValueType>
std::vector<SpatialOperator<ValueType>>
operator+(SpatialOperator<ValueType> a, SpatialOperator<ValueType> b)
{
    std::vector<SpatialOperator<ValueType>> result;
    result.push_back(std::move(a));
    result.push_back(std::move(b));
    return result;
}

/**
 * @brief Append a spatial operator to a vector.
 */
template<typename ValueType>
std::vector<SpatialOperator<ValueType>>
operator+(std::vector<SpatialOperator<ValueType>> ops, SpatialOperator<ValueType> op)
{
    ops.push_back(std::move(op));
    return ops;
}


/**
 * @class BlockExpression
 * @brief Per-equation expression that accumulates block spatial operators.
 *
 * Each BlockExpression represents one block-row of the coupled system
 * and records which operators contribute. Operators are routed to block
 * columns by field name.
 */
template<typename ValueType>
class BlockExpression
{

public:

    BlockExpression(localIdx equationIndex, std::vector<std::string> fieldNames)
        : equationIndex_(equationIndex), fieldNames_(std::move(fieldNames))
    {}

    BlockExpression& operator=(SpatialOperator<ValueType> op)
    {
        operators_.clear();
        operators_.push_back(std::move(op));
        return *this;
    }

    BlockExpression& operator=(std::vector<SpatialOperator<ValueType>> ops)
    {
        operators_ = std::move(ops);
        return *this;
    }

    [[nodiscard]] localIdx equationIndex() const { return equationIndex_; }

    [[nodiscard]] const std::vector<SpatialOperator<ValueType>>& operators() const
    {
        return operators_;
    }

    /**
     * @brief Find the column index of a field by name.
     * @return Column index in [0, nBlocks), or -1 if not found.
     */
    [[nodiscard]] localIdx fieldColumn(const std::string& name) const
    {
        for (localIdx j = 0; j < static_cast<localIdx>(fieldNames_.size()); ++j)
        {
            if (fieldNames_[static_cast<std::size_t>(j)] == name)
            {
                return j;
            }
        }
        return -1;
    }

private:

    localIdx equationIndex_;
    std::vector<std::string> fieldNames_;
    std::vector<SpatialOperator<ValueType>> operators_;
};


namespace imp
{

/**
 * @brief Create an implicit source term targeting a named field.
 * @param coeff Scalar coefficient.
 * @param field The field vector (unused for now, needed for future VolumeField overloads).
 * @param fieldName The name of the field this source couples to.
 */
SpatialOperator<scalar>
source(scalar coeff, const Vector<scalar>& field, const std::string& fieldName);

} // namespace imp

} // namespace NeoN::bdsl
