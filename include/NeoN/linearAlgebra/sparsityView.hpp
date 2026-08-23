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
     * @brief Sentinel storage offset returned by findEntry() when (i,j) is not stored.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr IndexType invalidIndex() { return std::numeric_limits<IndexType>::max(); }

    /**
     * @brief Retrieve the storage offset of the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Offset into colIdxs/values if it exists, invalidIndex() otherwise.
     * @note assumes colIdxs is sorted ascending within each row's range; not verified.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType findEntry(const IndexType i, const IndexType j) const
    {
        const IndexType nStored = rowSize(i);
        for (std::remove_const_t<IndexType> ic = 0; ic < nStored; ++ic)
        {
            const IndexType localCol = rowOffs[i] + ic;
            if (colIdxs[localCol] == j)
            {
                return localCol;
            }
            if (colIdxs[localCol] > j) break;
        }
        return invalidIndex();
    }

    /**
     * @brief Number of stored slots in row i. CSR has no padding, so every slot is a real
     * (logical, non-zero) entry -- unlike EllSparsityView::rowSize(), callers never need to
     * check colIdxs against invalidIndex() when walking a CSR row this way.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType rowSize(const IndexType i) const { return rowOffs[i + 1] - rowOffs[i]; }

    /**
     * @brief Flat storage offset of slot `slot` of row `i`.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType linearIndex(const IndexType i, const IndexType slot) const
    {
        return rowOffs[i] + slot;
    }

    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType entry(const IndexType i, const IndexType j) const
    {
        const IndexType offset = findEntry(i, j);
        if (offset == invalidIndex())
        {
            Kokkos::abort("Memory not allocated for CSR matrix component.");
        }
        return offset;
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
     * @brief Sentinel column index marking an unused (padding) slot. For a signed IndexType
     * this matches gko::invalid_index<IndexType>() (-1) -- Ginkgo's own ELL kernels skip padding
     * by checking a slot's column index against exactly this value, so this is what lets colIdxs
     * be handed to gko::matrix::Ell::create_const() unmodified. localIdx is signed unless built
     * with NeoN_US_IDX; for an unsigned IndexType this is still a valid NeoN-internal sentinel
     * (all bits set) but Ginkgo itself rejects unsigned index types outright.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr IndexType invalidIndex() { return static_cast<IndexType>(-1); }

    /**
     * @brief Flat storage offset of slot `slot` of row `i` (column-major layout).
     */
    KOKKOS_INLINE_FUNCTION
    IndexType linearIndex(const IndexType i, const IndexType slot) const
    {
        return i + stride * slot;
    }

    /**
     * @brief Number of stored slots in row i, i.e. the padded row width -- the same for every
     * row. Unlike SparsityView::rowSize() (CSR, no padding), callers walking a row via this must
     * still check colIdxs[linearIndex(i,slot)] against invalidIndex() and stop there, since a
     * row's real entry count can be less than this.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType rowSize(const IndexType) const { return numStoredElementsPerRow; }

    /**
     * @brief Retrieve the storage offset of the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Flat offset into colIdxs/values if it exists, invalidIndex() otherwise.
     * @note assumes each row's stored columns are sorted ascending with padding
     * (invalidIndex()) trailing; not verified.
     */
    KOKKOS_INLINE_FUNCTION
    IndexType findEntry(const IndexType i, const IndexType j) const
    {
        for (std::remove_const_t<IndexType> slot = 0; slot < numStoredElementsPerRow; ++slot)
        {
            const IndexType idx = linearIndex(i, static_cast<IndexType>(slot));
            const IndexType col = colIdxs[idx];
            if (col == j) return idx;
            if (col == invalidIndex() || col > j) break;
        }
        return invalidIndex();
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
        const IndexType offset = findEntry(i, j);
        if (offset == invalidIndex())
        {
            Kokkos::abort("Memory not allocated for ELL matrix component.");
        }
        return offset;
    }

    View<const IndexType> colIdxs;
    IndexType numStoredElementsPerRow;
    IndexType stride;
};

} // namespace NeoN::la
