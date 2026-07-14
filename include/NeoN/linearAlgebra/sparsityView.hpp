// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include "NeoN/core/view.hpp"

namespace NeoN::la
{

/**
 * @struct Dimensions
 * @brief hold the number of rows and columns of a matrix
 */
struct Dimensions
{
    localIdx rows;
    localIdx cols;
};

/**
 * @struct SparsityView
 * @brief A view struct to allow easy read/write on all executors.
 *
 * @tparam IndexType The index type of the rows and columns.
 * @todo ideally this should be immutable
 */
template<typename IndexType>
struct SparsityView
{
    SparsityView(View<const IndexType> colIdxsView, View<const IndexType> rowOffsView)
        : colIdxs(colIdxsView), rowOffs(rowOffsView) {};


    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType entry(const IndexType i, const IndexType j) const
    {
        const IndexType rowSize = rowOffs[i + 1] - rowOffs[i];
        for (std::remove_const_t<IndexType> ic = 0; ic < rowSize; ++ic)
        {
            const IndexType localCol = rowOffs[i] + ic;
            if (colIdxs[localCol] == j)
            {
                return localCol;
            }
            if (colIdxs[localCol] > j) break;
        }
        Kokkos::abort("Memory not allocated for CSR matrix component.");
        return 0; // compiler warning suppression.
    }

    View<const IndexType> colIdxs;
    View<const IndexType> rowOffs;
};

/**
 * @struct EllSparsityView
 * @brief A view struct to allow easy read/write on all executors for the ELLPACK
 * (fixed-width, padded) sparsity pattern.
 *
 * Column indices are stored column-major across the padded row slots, i.e. slot
 * `s` of row `i` lives at flat offset `i + stride * s`. Rows with fewer nonzeros
 * than `numStoredElementsPerRow` are padded with `invalidIndex()`.
 *
 * @tparam IndexType The index type of the rows and columns.
 * @todo ideally this should be immutable
 */
template<typename IndexType>
struct EllSparsityView
{
    EllSparsityView(
        View<const IndexType> colIdxsView, IndexType numStoredElementsPerRowIn, IndexType strideIn
    )
        : colIdxs(colIdxsView), numStoredElementsPerRow(numStoredElementsPerRowIn),
          stride(strideIn) {};

    /**
     * @brief Sentinel column index marking an unused (padding) slot.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr IndexType invalidIndex() { return std::numeric_limits<IndexType>::max(); }

    /**
     * @brief Flat storage offset of slot `slot` of row `i` (column-major layout).
     */
    KOKKOS_INLINE_FUNCTION
    IndexType linearIndex(const IndexType i, const IndexType slot) const
    {
        return i + stride * slot;
    }

    /**
     * @brief Retrieve the storage offset of the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Flat offset into colIdxs/values if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType entry(const IndexType i, const IndexType j) const
    {
        for (std::remove_const_t<IndexType> slot = 0; slot < numStoredElementsPerRow; ++slot)
        {
            const IndexType idx = linearIndex(i, static_cast<IndexType>(slot));
            const IndexType col = colIdxs[idx];
            if (col == j) return idx;
            if (col == invalidIndex() || col > j) break;
        }
        Kokkos::abort("Memory not allocated for ELL matrix component.");
        return 0; // compiler warning suppression.
    }

    View<const IndexType> colIdxs;
    IndexType numStoredElementsPerRow;
    IndexType stride;
};

} // namespace NeoN::la
