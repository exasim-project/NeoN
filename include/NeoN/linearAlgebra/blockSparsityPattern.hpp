// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

/**
 * @class BlockSparsityPattern
 * @brief Monolithic sparsity pattern expanded from a base per-block sparsity.
 *
 * Given nBlocks and a base sparsity (nCells rows, baseNnz non-zeros), computes
 * the expanded monolithic pattern with nBlocks*nCells rows and nBlocks^2*baseNnz
 * non-zeros. The expansion only depends on nBlocks + base sparsity, so it is
 * computed once at construction.
 */
class BlockSparsityPattern : public SparsityPattern<localIdx>
{

public:

    /**
     * @brief Construct from nBlocks and base sparsity. Computes expanded colIdxs/rowOffs.
     */
    BlockSparsityPattern(localIdx nBlocks, const SparsityPattern<localIdx>& baseSparsity);

    /**
     * @brief Copy constructor (copies base SparsityPattern + metadata).
     */
    BlockSparsityPattern(const BlockSparsityPattern& other);

    /** @brief Number of blocks per dimension. */
    [[nodiscard]] localIdx nBlocks() const;

    /** @brief Number of cells (rows in each inner block). */
    [[nodiscard]] localIdx nCells() const;

    /** @brief Number of non-zeros per inner block. */
    [[nodiscard]] localIdx baseNnz() const;

    /** @brief Copy the pattern to host (preserving block metadata). */
    [[nodiscard]] BlockSparsityPattern copyToHost() const;

    /** @brief Copy the pattern to a destination executor (preserving block metadata). */
    [[nodiscard]] BlockSparsityPattern copyToExecutor(Executor dstExec) const;

private:

    // Private constructor for copyToHost/copyToExecutor
    BlockSparsityPattern(
        localIdx nBlocks,
        localIdx nCells,
        localIdx baseNnz,
        Vector<localIdx>&& colIdxs,
        Vector<localIdx>&& rowOffs
    );

    localIdx nBlocks_;
    localIdx nCells_;
    localIdx baseNnz_;
};

/**
 * @brief A CSR matrix whose sparsity is a block-expanded pattern.
 */
using BlockCSRMatrix = Matrix<scalar, BlockSparsityPattern>;

/**
 * @brief Convert a BlockCSRMatrix to a plain CSRMatrix by upcasting the sparsity pointer.
 *
 * The returned CSRMatrix shares the same underlying data (no deep copy).
 */
CSRMatrix<scalar, localIdx> toCSR(const BlockCSRMatrix& bm);


} // namespace NeoN::la
