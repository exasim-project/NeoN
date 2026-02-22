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
 * @brief Device-safe view into an nBlocks x nBlocks dense coupling matrix.
 *
 * Represents the coupling between field components at a single CSR position
 * (cell pair). Stored in column-major order: operator()(i, j) accesses
 * values[i + j * nBlocks].
 */
struct BlockView
{
    View<scalar> values; ///< Column-major data, size = nBlocks * nBlocks
    localIdx nBlocks;

    /**
     * @brief 2D access into the coupling matrix.
     * @param i Row (field component row).
     * @param j Column (field component column).
     * @return Reference to the coupling value.
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator()(localIdx i, localIdx j) const { return values[i + j * nBlocks]; }

    /**
     * @brief Flat offset access.
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator[](localIdx offset) const { return values[offset]; }
};

/**
 * @struct BlockRowView
 * @brief Device-safe view into selected rows of a coupling matrix.
 *
 * Returned by BlockMatrixView::rowView(k, startRow, endRow). Represents a
 * rectangular nRows x nBlocks sub-matrix of the coupling matrix at CSR
 * position k. The underlying column-major stride is nBlocks (the full
 * coupling matrix height), so operator()(i, j) = values[i + j * nBlocks].
 */
struct BlockRowView
{
    View<scalar> values; ///< Subview starting at startRow within the coupling matrix
    localIdx nBlocks;    ///< Number of columns (and column-major stride)
    localIdx nRows;      ///< Number of selected rows (endRow - startRow)

    /**
     * @brief 2D access into the rectangular sub-matrix.
     * @param i Row index (0 .. nRows - 1).
     * @param j Column index (0 .. nBlocks - 1).
     * @return Reference to the coupling value.
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator()(localIdx i, localIdx j) const { return values[i + j * nBlocks]; }
};

/**
 * @struct BlockMatrixView
 * @brief Device-safe view into the full block matrix structure.
 *
 * Values are interleaved by CSR position: at each of the nnz non-zero
 * positions, an nBlocks x nBlocks column-major coupling matrix is stored.
 * operator()(k) returns the BlockView at CSR position k.
 * rowView(k, startRow, endRow) returns a rectangular BlockRowView.
 */
struct BlockMatrixView
{
    SparsityView<localIdx> sparsity;
    View<scalar> allValues; ///< Size = nnz * nBlocks^2
    localIdx nBlocks;
    localIdx nCells; ///< Number of cells (rows in the sparsity pattern)
    localIdx nnz;    ///< Total number of non-zeros in the sparsity pattern

    /**
     * @brief Access the coupling matrix at CSR position k.
     * @param k Global CSR non-zero index.
     * @return BlockView for the nBlocks x nBlocks coupling matrix.
     */
    KOKKOS_INLINE_FUNCTION
    BlockView operator()(localIdx k) const
    {
        localIdx nb2 = nBlocks * nBlocks;
        return BlockView {allValues.subview(k * nb2, nb2), nBlocks};
    }

    /**
     * @brief View into selected rows of the coupling matrix at CSR position k.
     * @param k Global CSR non-zero index.
     * @param startRow First row (inclusive).
     * @param endRow Last row (exclusive).
     * @return BlockRowView for the nRows x nBlocks rectangular sub-matrix.
     */
    KOKKOS_INLINE_FUNCTION
    BlockRowView rowView(localIdx k, localIdx startRow, localIdx endRow) const
    {
        localIdx nb2 = nBlocks * nBlocks;
        localIdx nRows = endRow - startRow;
        localIdx viewSize = (nBlocks - 1) * nBlocks + nRows;
        return BlockRowView {allValues.subview(k * nb2 + startRow, viewSize), nBlocks, nRows};
    }
};

} // namespace NeoN::la
