// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/blockMatrixView.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

/**
 * @class BlockMatrix
 * @brief A block-structured sparse matrix with shared sparsity across all blocks.
 *
 * Values are interleaved by CSR position: at each of the nnz non-zero positions,
 * an nBlocks x nBlocks column-major coupling matrix is stored. Total values size
 * is nnz * nBlocks^2.
 */
class BlockMatrix
{

public:

    /**
     * @brief Construct from nBlocks and shared sparsity. Values are zero-initialized.
     */
    BlockMatrix(
        const Executor& exec, localIdx nBlocks, std::shared_ptr<SparsityPattern<localIdx>> sparsity
    );

    /**
     * @brief Construct with pre-built values (size must be nBlocks^2 * nnz).
     */
    BlockMatrix(
        const Executor& exec,
        localIdx nBlocks,
        std::shared_ptr<SparsityPattern<localIdx>> sparsity,
        const Vector<scalar>& values
    );

    /** @brief Number of blocks per dimension. */
    [[nodiscard]] localIdx nBlocks() const;

    /** @brief Number of cells (rows in each inner block). */
    [[nodiscard]] localIdx nCells() const;

    /** @brief Number of non-zeros per inner block. */
    [[nodiscard]] localIdx nnz() const;

    /** @brief Total number of monolithic rows (nBlocks * nCells). */
    [[nodiscard]] localIdx totalSize() const;

    /** @brief Access the shared sparsity pattern. */
    [[nodiscard]] const SparsityPattern<localIdx>& sparsity() const;

    /** @brief Access the flat values vector. */
    [[nodiscard]] Vector<scalar>& values();

    /** @brief Access the flat values vector (const). */
    [[nodiscard]] const Vector<scalar>& values() const;

    /** @brief Device-safe view (only on lvalues). */
    [[nodiscard]] BlockMatrixView view() &;

    /** @brief Prevent view() on temporaries. */
    BlockMatrixView view() && = delete;

    /** @brief Get the executor. */
    [[nodiscard]] const Executor& exec() const;

private:

    Executor exec_;
    localIdx nBlocks_;
    std::shared_ptr<SparsityPattern<localIdx>> sparsity_;
    Vector<scalar> values_; ///< Size = nnz * nBlocks^2 (interleaved by CSR position)
};

} // namespace NeoN::la
