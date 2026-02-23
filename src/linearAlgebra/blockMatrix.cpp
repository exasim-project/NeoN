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
      blockMatrix_(
          Vector<scalar>(exec, nBlocks * nBlocks * sparsity_->nnz(), 0.0),
          std::make_shared<const BlockSparsityPattern>(nBlocks, *sparsity_)
      )
{}

BlockMatrix::BlockMatrix(
    const Executor& exec,
    localIdx nBlocks,
    std::shared_ptr<SparsityPattern<localIdx>> sparsity,
    const Vector<scalar>& values
)
    : exec_(exec), nBlocks_(nBlocks), sparsity_(std::move(sparsity)),
      blockMatrix_(values, std::make_shared<const BlockSparsityPattern>(nBlocks, *sparsity_))
{}

localIdx BlockMatrix::nBlocks() const { return nBlocks_; }

localIdx BlockMatrix::nCells() const { return sparsity_->rows(); }

localIdx BlockMatrix::nnz() const { return sparsity_->nnz(); }

localIdx BlockMatrix::totalSize() const { return nBlocks_ * sparsity_->rows(); }

const SparsityPattern<localIdx>& BlockMatrix::sparsity() const { return *sparsity_; }

Vector<scalar>& BlockMatrix::values() { return blockMatrix_.values(); }

const Vector<scalar>& BlockMatrix::values() const { return blockMatrix_.values(); }

const BlockCSRMatrix& BlockMatrix::blockCSRMatrix() const { return blockMatrix_; }

const Executor& BlockMatrix::exec() const { return exec_; }

BlockMatrixView BlockMatrix::view() &
{
    return BlockMatrixView {
        sparsity_->view(),
        blockMatrix_.values().view(),
        nBlocks_,
        sparsity_->rows(),
        sparsity_->nnz()
    };
}


} // namespace NeoN::la
