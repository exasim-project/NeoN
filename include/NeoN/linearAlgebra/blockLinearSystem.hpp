// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/blockMatrix.hpp"
#include "NeoN/linearAlgebra/blockVector.hpp"

namespace NeoN::la
{

/**
 * @struct SourceTerm
 * @brief Lightweight host-side term: coefficient * field.
 */
struct SourceTerm
{
    scalar coefficient;
    Vector<scalar>* field;
};

using SourceExpression = std::vector<SourceTerm>;

/**
 * @brief Create a source term from a coefficient and field reference.
 */
SourceTerm source(scalar coeff, Vector<scalar>& field);

/**
 * @brief Combine two source terms into an expression.
 */
SourceExpression operator+(const SourceTerm& a, const SourceTerm& b);

/**
 * @brief Append a source term to an existing expression.
 */
SourceExpression operator+(SourceExpression terms, const SourceTerm& t);

/**
 * @class BlockExpression
 * @brief Per-equation expression that accumulates source terms.
 *
 * Each BlockExpression represents one block-row of the coupled system
 * and records which fields contribute (and with what coefficient).
 */
class BlockExpression
{

public:

    BlockExpression(localIdx equationIndex, const std::vector<Vector<scalar>*>& fields);

    BlockExpression& operator=(const SourceTerm& term);
    BlockExpression& operator=(const SourceExpression& terms);

    [[nodiscard]] localIdx equationIndex() const;
    [[nodiscard]] const SourceExpression& terms() const;

    /**
     * @brief Find the column index of a field pointer (linear search over fields).
     * @return Column index in [0, nBlocks), or -1 if not found.
     */
    [[nodiscard]] localIdx fieldIndex(const Vector<scalar>* field) const;

private:

    localIdx equationIndex_;
    std::vector<Vector<scalar>*> fields_;
    SourceExpression terms_;
};

/**
 * @class BlockLinearSystem
 * @brief Orchestrator that owns a block matrix, RHS, and solution vectors.
 *
 * Ties together BlockMatrix, BlockVector, and BlockSolve with expression-driven
 * assembly using simple source(coeff, field) terms. This is a minimal first
 * integration: only host-side source() terms (diagonal coupling entries).
 */
class BlockLinearSystem
{

public:

    /**
     * @brief Construct from executor, field pointers, sparsity, and solver config.
     */
    BlockLinearSystem(
        const Executor& exec,
        std::vector<Vector<scalar>*> fields,
        std::shared_ptr<SparsityPattern<localIdx>> sparsity,
        const Dictionary& solverDict
    );

    /**
     * @brief Access both expressions for 2-field systems (structured binding).
     */
    std::pair<BlockExpression&, BlockExpression&> expressions();

    /**
     * @brief Access the i-th expression (general N-field access).
     */
    BlockExpression& expression(localIdx i);

    /**
     * @brief Set the RHS vector for equation i.
     */
    void setRhs(localIdx i, const Vector<scalar>& rhs);

    /**
     * @brief Assemble source terms into block matrix coupling entries.
     */
    void assemble();

    /**
     * @brief Gather fields → solve → scatter back to fields.
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
    std::vector<BlockExpression> expressions_;
};

} // namespace NeoN::la
