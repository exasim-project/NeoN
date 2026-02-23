// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockLinearSystem.hpp"
#include "NeoN/linearAlgebra/blockSolver.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::la
{

// -- Free functions -----------------------------------------------------------

SourceTerm source(scalar coeff, Vector<scalar>& field) { return SourceTerm {coeff, &field}; }

SourceExpression operator+(const SourceTerm& a, const SourceTerm& b)
{
    return SourceExpression {a, b};
}

SourceExpression operator+(SourceExpression terms, const SourceTerm& t)
{
    terms.push_back(t);
    return terms;
}

// -- BlockExpression ----------------------------------------------------------

BlockExpression::BlockExpression(localIdx equationIndex, const std::vector<Vector<scalar>*>& fields)
    : equationIndex_(equationIndex), fields_(fields), terms_()
{}

BlockExpression& BlockExpression::operator=(const SourceTerm& term)
{
    terms_ = {term};
    return *this;
}

BlockExpression& BlockExpression::operator=(const SourceExpression& terms)
{
    terms_ = terms;
    return *this;
}

localIdx BlockExpression::equationIndex() const { return equationIndex_; }

const SourceExpression& BlockExpression::terms() const { return terms_; }

localIdx BlockExpression::fieldIndex(const Vector<scalar>* field) const
{
    for (localIdx j = 0; j < static_cast<localIdx>(fields_.size()); ++j)
    {
        if (fields_[static_cast<std::size_t>(j)] == field)
        {
            return j;
        }
    }
    return -1;
}

// -- BlockLinearSystem --------------------------------------------------------

BlockLinearSystem::BlockLinearSystem(
    const Executor& exec,
    std::vector<Vector<scalar>*> fields,
    std::shared_ptr<SparsityPattern<localIdx>> sparsity,
    const Dictionary& solverDict
)
    : exec_(exec), fields_(std::move(fields)), sparsity_(std::move(sparsity)),
      solverDict_(solverDict), matrix_(exec_, static_cast<localIdx>(fields_.size()), sparsity_),
      rhs_(exec_, static_cast<localIdx>(fields_.size()), sparsity_->rows()),
      solution_(exec_, static_cast<localIdx>(fields_.size()), sparsity_->rows())
{
    localIdx nFields = static_cast<localIdx>(fields_.size());
    expressions_.reserve(static_cast<std::size_t>(nFields));
    for (localIdx i = 0; i < nFields; ++i)
    {
        expressions_.emplace_back(i, fields_);
    }
}

std::pair<BlockExpression&, BlockExpression&> BlockLinearSystem::expressions()
{
    return {expressions_[0], expressions_[1]};
}

BlockExpression& BlockLinearSystem::expression(localIdx i)
{
    return expressions_[static_cast<std::size_t>(i)];
}

void BlockLinearSystem::setRhs(localIdx i, const Vector<scalar>& rhs)
{
    rhs_.copyBlockFrom(i, rhs);
}

void BlockLinearSystem::assemble()
{
    // Zero out matrix values
    fill(matrix_.values(), 0.0);

    auto spView = sparsity_->view();
    auto bmView = matrix_.view();
    localIdx nCells = sparsity_->rows();

    for (const auto& expr : expressions_)
    {
        localIdx eqI = expr.equationIndex();
        for (const auto& term : expr.terms())
        {
            localIdx colJ = expr.fieldIndex(term.field);
            scalar coeff = term.coefficient;

            parallelFor(
                exec_,
                {0, nCells},
                NEON_LAMBDA(const localIdx celli) {
                    localIdx k = spView.entry(celli, celli);
                    bmView(k)(eqI, colJ) += coeff;
                },
                "blockLinearSystem_assemble"
            );
        }
    }
}

void BlockLinearSystem::solve()
{
    localIdx nFields = static_cast<localIdx>(fields_.size());

    // Gather: copy each field into the solution block vector
    for (localIdx i = 0; i < nFields; ++i)
    {
        solution_.copyBlockFrom(i, *fields_[static_cast<std::size_t>(i)]);
    }

    BlockSolver solver(exec_, solverDict_);
    solver.solve(matrix_, rhs_, solution_);

    // Scatter: copy solution blocks back to fields
    for (localIdx i = 0; i < nFields; ++i)
    {
        solution_.copyBlockTo(i, *fields_[static_cast<std::size_t>(i)]);
    }
}

const BlockMatrix& BlockLinearSystem::matrix() const { return matrix_; }

const BlockVector& BlockLinearSystem::rhs() const { return rhs_; }

} // namespace NeoN::la
