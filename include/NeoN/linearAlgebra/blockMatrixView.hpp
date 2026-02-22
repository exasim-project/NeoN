// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

/**
 * @struct BlockView
 * @brief Device-safe, mdspan-like view into a single CSR block.
 *
 * Wraps a values slice and a shared sparsity pattern, providing 2D element
 * access via operator()(row, col) and direct offset access via operator[].
 */
struct BlockView
{
    View<scalar> values;
    SparsityView<localIdx> sparsity;

    /**
     * @brief 2D element access (mdspan-like).
     * @param i Row index within this block.
     * @param j Column index within this block.
     * @return Reference to the scalar value at (i, j).
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator()(localIdx i, localIdx j) const { return values[sparsity.entry(i, j)]; }

    /**
     * @brief Direct offset access into the values array.
     * @param offset The flat offset into the block's values.
     * @return Reference to the scalar value.
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator[](localIdx offset) const { return values[offset]; }
};

/**
 * @struct BlockRowView
 * @brief Device-safe view into a single block row.
 *
 * Returned by BlockMatrixView::row(I). An expression assembles into exactly
 * one BlockRowView. operator()(J) computes the BlockView for block (I, J)
 * on-the-fly from the flat values array.
 */
struct BlockRowView
{
    SparsityView<localIdx> sparsity;
    View<scalar> allValues; ///< Full flat values array
    localIdx rowIndex;      ///< Block row I
    localIdx nBlocks;
    localIdx nnz; ///< Non-zeros per block in the shared sparsity

    /**
     * @brief Access block column J in this row as a BlockView.
     */
    KOKKOS_INLINE_FUNCTION
    BlockView operator()(localIdx j) const
    {
        localIdx offset = (rowIndex * nBlocks + j) * nnz;
        return BlockView {allValues.subview(offset, nnz), sparsity};
    }
};

/**
 * @struct BlockMatrixView
 * @brief Device-safe view into the full block matrix structure.
 *
 * Provides (I, J) -> BlockView lookup, row extraction, and global entry access.
 * All nBlocks^2 blocks share a single sparsity pattern. Block (I, J) occupies
 * values slice [(I * nBlocks + J) * nnz, ... + nnz).
 */
struct BlockMatrixView
{
    SparsityView<localIdx> sparsity;
    View<scalar> allValues; ///< Size = nBlocks^2 * nnz
    localIdx nBlocks;
    localIdx nCells; ///< Number of cells (rows in each inner block)
    localIdx nnz;    ///< Non-zeros per block

    /**
     * @brief Access block (I, J) as a BlockView (computed on-the-fly).
     */
    KOKKOS_INLINE_FUNCTION
    BlockView operator()(localIdx i, localIdx j) const
    {
        localIdx offset = (i * nBlocks + j) * nnz;
        return BlockView {allValues.subview(offset, nnz), sparsity};
    }

    /**
     * @brief Extract block row I as a BlockRowView.
     */
    KOKKOS_INLINE_FUNCTION
    BlockRowView row(localIdx i) const
    {
        return BlockRowView {sparsity, allValues, i, nBlocks, nnz};
    }

    /**
     * @brief Global (row, col) access — routes to the correct block.
     * @param row Global row index (0 .. nBlocks * nCells - 1).
     * @param col Global column index (0 .. nBlocks * nCells - 1).
     */
    KOKKOS_INLINE_FUNCTION
    scalar& entry(localIdx row, localIdx col) const
    {
        localIdx I = row / nCells;
        localIdx J = col / nCells;
        localIdx localRow = row - I * nCells;
        localIdx localCol = col - J * nCells;
        localIdx offset = (I * nBlocks + J) * nnz;
        auto blockValues = allValues.subview(offset, nnz);
        return blockValues[sparsity.entry(localRow, localCol)];
    }
};

} // namespace NeoN::la
