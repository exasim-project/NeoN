// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/view.hpp"

namespace NeoN::la
{

/**
 * @struct dimensions
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

} // namespace NeoN::la
