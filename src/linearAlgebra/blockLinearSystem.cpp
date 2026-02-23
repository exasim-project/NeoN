// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockLinearSystem.hpp"
#include "NeoN/linearAlgebra/blockSolver.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"

namespace NeoN::la
{

// -- BlockLinearSystem --------------------------------------------------------

BlockLinearSystem::BlockLinearSystem(
    const Executor& exec,
    std::vector<std::string> fieldNames,
    std::vector<Vector<scalar>*> fields,
    std::shared_ptr<SparsityPattern<localIdx>> sparsity,
    const Dictionary& solverDict
)
    : exec_(exec), fields_(std::move(fields)), sparsity_(std::move(sparsity)),
      solverDict_(solverDict), matrix_(exec_, static_cast<localIdx>(fields_.size()), sparsity_),
      rhs_(exec_, static_cast<localIdx>(fields_.size()), sparsity_->rows()),
      solution_(exec_, static_cast<localIdx>(fields_.size()), sparsity_->rows()),
      fieldNames_(std::move(fieldNames))
{
    localIdx nFields = static_cast<localIdx>(fields_.size());
    expressions_.reserve(static_cast<std::size_t>(nFields));
    for (localIdx i = 0; i < nFields; ++i)
    {
        expressions_.emplace_back(i, fieldNames_);
    }
}

bdsl::BlockExpression<scalar>& BlockLinearSystem::expression(localIdx i)
{
    return expressions_[static_cast<std::size_t>(i)];
}

void BlockLinearSystem::setRhs(localIdx i, const Vector<scalar>& rhs)
{
    rhs_.copyBlockFrom(i, rhs);
}

void BlockLinearSystem::assemble()
{
    fill(matrix_.values(), 0.0);

    auto spView = sparsity_->view();
    auto bmView = matrix_.view();
    localIdx nCells = sparsity_->rows();

    for (const auto& expr : expressions_)
    {
        localIdx eqI = expr.equationIndex();
        for (const auto& op : expr.operators())
        {
            localIdx colJ = expr.fieldColumn(op.getFieldName());
            op.implicitOperation(bmView, spView, eqI, colJ, nCells, exec_);
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
