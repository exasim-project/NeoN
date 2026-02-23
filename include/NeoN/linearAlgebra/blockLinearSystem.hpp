// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/blockDsl.hpp"
#include "NeoN/linearAlgebra/blockMatrix.hpp"
#include "NeoN/linearAlgebra/blockVector.hpp"

namespace NeoN::la
{

/**
 * @class BlockLinearSystem
 * @brief Orchestrator that owns a block matrix, RHS, and solution vectors.
 *
 * Ties together BlockMatrix, BlockVector, and BlockSolve with expression-driven
 * assembly using bdsl operators. Operators carry field names; BlockLinearSystem
 * maps names to block columns during assembly.
 */
class BlockLinearSystem
{

public:

    /**
     * @brief Construct with named fields for bdsl-based assembly.
     */
    BlockLinearSystem(
        const Executor& exec,
        std::vector<std::string> fieldNames,
        std::vector<Vector<scalar>*> fields,
        std::shared_ptr<SparsityPattern<localIdx>> sparsity,
        const Dictionary& solverDict
    );

    /**
     * @brief Access N expressions for structured binding.
     */
    template<std::size_t N>
    auto expressions()
    {
        return [this]<std::size_t... I>(std::index_sequence<I...>)
        { return std::forward_as_tuple(expressions_[I]...); }(std::make_index_sequence<N> {});
    }

    /**
     * @brief Access the i-th expression.
     */
    bdsl::BlockExpression<scalar>& expression(localIdx i);

    /**
     * @brief Set the RHS vector for equation i.
     */
    void setRhs(localIdx i, const Vector<scalar>& rhs);

    /**
     * @brief Assemble operators into block matrix coupling entries.
     */
    void assemble();

    /**
     * @brief Gather fields, solve, scatter back to fields.
     */
    void solve();

    [[nodiscard]] const BlockMatrix& matrix() const;
    [[nodiscard]] const BlockVector& rhs() const;

private:

    Executor exec_;
    std::vector<Vector<scalar>*> fields_;
    std::shared_ptr<SparsityPattern<localIdx>> sparsity_;
    Dictionary solverDict_;
    BlockMatrix matrix_;
    BlockVector rhs_;
    BlockVector solution_;
    std::vector<std::string> fieldNames_;
    std::vector<bdsl::BlockExpression<scalar>> expressions_;
};

} // namespace NeoN::la
