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

CSRMatrix<scalar, localIdx> BlockMatrix::monolithic() const
{
    // Copy sparsity and values to host for assembly
    auto hostSp = sparsity_->copyToHost();
    auto hostVals = values_.copyToHost();

    auto spColIdxs = hostSp.colIdxs().view();
    auto spRowOffs = hostSp.rowOffs().view();
    auto blockVals = hostVals.view();

    localIdx nc = hostSp.rows();
    localIdx innerNnz = hostSp.nnz();
    localIdx monoRows = nBlocks_ * nc;

    // Count non-zeros per monolithic row:
    // each monolithic row r = I * nc + localRow has nBlocks * (nnzInLocalRow) entries
    std::vector<localIdx> monoRowOffsVec(static_cast<size_t>(monoRows + 1), 0);
    for (localIdx I = 0; I < nBlocks_; ++I)
    {
        for (localIdx localRow = 0; localRow < nc; ++localRow)
        {
            localIdx r = I * nc + localRow;
            localIdx rowNnz = spRowOffs[localRow + 1] - spRowOffs[localRow];
            monoRowOffsVec[static_cast<size_t>(r + 1)] = nBlocks_ * rowNnz;
        }
    }

    // Prefix sum to get row offsets
    for (localIdx r = 0; r < monoRows; ++r)
    {
        monoRowOffsVec[static_cast<size_t>(r + 1)] += monoRowOffsVec[static_cast<size_t>(r)];
    }

    localIdx monoNnz = monoRowOffsVec[static_cast<size_t>(monoRows)];
    std::vector<localIdx> monoColIdxsVec(static_cast<size_t>(monoNnz));
    std::vector<scalar> monoValsVec(static_cast<size_t>(monoNnz));

    // Fill values and column indices
    for (localIdx I = 0; I < nBlocks_; ++I)
    {
        for (localIdx localRow = 0; localRow < nc; ++localRow)
        {
            localIdx r = I * nc + localRow;
            localIdx writePos = monoRowOffsVec[static_cast<size_t>(r)];

            for (localIdx J = 0; J < nBlocks_; ++J)
            {
                localIdx blockOffset = (I * nBlocks_ + J) * innerNnz;
                for (localIdx k = spRowOffs[localRow]; k < spRowOffs[localRow + 1]; ++k)
                {
                    monoColIdxsVec[static_cast<size_t>(writePos)] = J * nc + spColIdxs[k];
                    monoValsVec[static_cast<size_t>(writePos)] = blockVals[blockOffset + k];
                    ++writePos;
                }
            }
        }
    }

    // Build on host, then copy to target executor
    SerialExecutor hostExec;
    Vector<scalar> monoValues(hostExec, monoValsVec);
    Vector<localIdx> monoColIdxs(hostExec, monoColIdxsVec);
    Vector<localIdx> monoRowOffs(hostExec, monoRowOffsVec);

    auto result = CSRMatrix<scalar, localIdx>(monoValues, monoColIdxs, monoRowOffs);
    return result.copyToExecutor(exec_);
}

} // namespace NeoN::la
