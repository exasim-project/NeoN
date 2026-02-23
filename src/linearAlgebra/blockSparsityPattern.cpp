// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockSparsityPattern.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::la
{

namespace
{

SparsityPattern<localIdx>
buildBlockSparsity(localIdx nBlocks, const SparsityPattern<localIdx>& baseSparsity)
{
    const Executor& exec = baseSparsity.exec();
    const localIdx nb = nBlocks;
    const localIdx nc = baseSparsity.rows();
    const localIdx monoRows = nb * nc;
    const localIdx monoNnz = nb * nb * baseSparsity.nnz();

    auto baseRowView = baseSparsity.rowOffs().view();

    // Parallel exclusive prefix sum for monolithic row offsets
    Vector<localIdx> monoRowOffs(exec, monoRows + 1, 0);
    auto monoRowView = monoRowOffs.view();

    parallelScan(
        exec,
        {0, monoRows},
        NEON_LAMBDA(const localIdx idx, localIdx& update, const bool finalPass) {
            const localIdx localRow = idx % nc;
            localIdx nnzInRow = nb * (baseRowView[localRow + 1] - baseRowView[localRow]);
            if (finalPass)
            {
                monoRowView[idx] = update;
            }
            update += nnzInRow;
            if (finalPass && idx == monoRows - 1)
            {
                monoRowView[monoRows] = update;
            }
        }
    );

    // Fill column indices in parallel
    Vector<localIdx> monoColIdxs(exec, monoNnz, 0);
    auto baseColView = baseSparsity.colIdxs().view();
    auto monoColView = monoColIdxs.view();

    parallelFor(
        exec,
        {0, monoRows},
        NEON_LAMBDA(const localIdx monoRow) {
            const localIdx I = monoRow / nc;
            const localIdx localRow = monoRow % nc;
            localIdx pos = monoRowView[monoRow];
            for (localIdx J = 0; J < nb; ++J)
            {
                for (localIdx k = baseRowView[localRow]; k < baseRowView[localRow + 1]; ++k)
                {
                    monoColView[pos] = J * nc + baseColView[k];
                    ++pos;
                }
            }
        },
        "buildBlockSparsity"
    );

    return SparsityPattern<localIdx>(std::move(monoColIdxs), std::move(monoRowOffs));
}

} // anonymous namespace

BlockSparsityPattern::BlockSparsityPattern(
    localIdx nBlocks, const SparsityPattern<localIdx>& baseSparsity
)
    : SparsityPattern<localIdx>(buildBlockSparsity(nBlocks, baseSparsity)), nBlocks_(nBlocks),
      nCells_(baseSparsity.rows()), baseNnz_(baseSparsity.nnz())
{}

BlockSparsityPattern::BlockSparsityPattern(const BlockSparsityPattern& other)
    : SparsityPattern<localIdx>(other), nBlocks_(other.nBlocks_), nCells_(other.nCells_),
      baseNnz_(other.baseNnz_)
{}

BlockSparsityPattern::BlockSparsityPattern(
    localIdx nBlocks,
    localIdx nCells,
    localIdx baseNnz,
    Vector<localIdx>&& colIdxs,
    Vector<localIdx>&& rowOffs
)
    : SparsityPattern<localIdx>(std::move(colIdxs), std::move(rowOffs)), nBlocks_(nBlocks),
      nCells_(nCells), baseNnz_(baseNnz)
{}

localIdx BlockSparsityPattern::nBlocks() const { return nBlocks_; }

localIdx BlockSparsityPattern::nCells() const { return nCells_; }

localIdx BlockSparsityPattern::baseNnz() const { return baseNnz_; }

BlockSparsityPattern BlockSparsityPattern::copyToHost() const
{
    return BlockSparsityPattern(
        nBlocks_,
        nCells_,
        baseNnz_,
        colIdxs().copyToExecutor(SerialExecutor()),
        rowOffs().copyToExecutor(SerialExecutor())
    );
}

BlockSparsityPattern BlockSparsityPattern::copyToExecutor(Executor dstExec) const
{
    return BlockSparsityPattern(
        nBlocks_,
        nCells_,
        baseNnz_,
        colIdxs().copyToExecutor(dstExec),
        rowOffs().copyToExecutor(dstExec)
    );
}

CSRMatrix<scalar, localIdx> toCSR(const BlockCSRMatrix& bm)
{
    auto blockSparsity = bm.sparsity();
    auto baseSparsity = std::static_pointer_cast<const SparsityPattern<localIdx>>(blockSparsity);
    return CSRMatrix<scalar, localIdx>(bm.values(), baseSparsity);
}

} // namespace NeoN::la
