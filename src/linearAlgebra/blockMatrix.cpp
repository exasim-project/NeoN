// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockMatrix.hpp"

namespace NeoN::la
{

BlockMatrix::BlockMatrix(
    const Executor& exec, localIdx nBlocks, std::shared_ptr<SparsityPattern<localIdx>> sparsity
)
    : exec_(exec), nBlocks_(nBlocks), sparsity_(std::move(sparsity)),
      values_(exec, nBlocks * nBlocks * sparsity_->nnz(), 0.0)
{}

BlockMatrix::BlockMatrix(
    const Executor& exec,
    localIdx nBlocks,
    std::shared_ptr<SparsityPattern<localIdx>> sparsity,
    const Vector<scalar>& values
)
    : exec_(exec), nBlocks_(nBlocks), sparsity_(std::move(sparsity)), values_(values)
{}

localIdx BlockMatrix::nBlocks() const { return nBlocks_; }

localIdx BlockMatrix::nCells() const { return sparsity_->rows(); }

localIdx BlockMatrix::nnz() const { return sparsity_->nnz(); }

localIdx BlockMatrix::totalSize() const { return nBlocks_ * sparsity_->rows(); }

const SparsityPattern<localIdx>& BlockMatrix::sparsity() const { return *sparsity_; }

Vector<scalar>& BlockMatrix::values() { return values_; }

const Vector<scalar>& BlockMatrix::values() const { return values_; }

const Executor& BlockMatrix::exec() const { return exec_; }

BlockMatrixView BlockMatrix::view() &
{
    return BlockMatrixView {
        sparsity_->view(), values_.view(), nBlocks_, sparsity_->rows(), sparsity_->nnz()
    };
}

} // namespace NeoN::la
