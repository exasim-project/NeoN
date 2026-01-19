// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{

/**
 * @struct MatrixView
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

/* @class SparsityPattern
 * @brief row and column index representation of a mesh
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 *
 */
template<typename IndexType>
class SparsityPattern
{

    void validate() const;

public:

    using SparsityIndexType = IndexType;

    /* @brief create an "empty" SparsityPattern with a given size  */
    SparsityPattern(const SparsityPattern& sp);

    SparsityPattern(Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs);

    [[nodiscard]] SparsityPattern copyToHost() const
    {
        return SparsityPattern<IndexType>(colIdxs_.copyToHost(), rowOffs_.copyToHost());
    }

    ~SparsityPattern() = default;

    /*@brief getter for diagOffset */
    const Executor& exec() const { return rowOffs_.exec(); }

    /*@brief getter for colIdxs */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] Vector<IndexType>& colIdxs() { return colIdxs_; };

    /*@brief getter for rowOffs */
    [[nodiscard]] const Vector<IndexType>& rowOffs() const { return rowOffs_; };

    /*@brief getter for rowOffs */
    [[nodiscard]] Vector<IndexType>& rowOffs() { return rowOffs_; };

    [[nodiscard]] localIdx rows() const { return rowOffs_.size() - 1; };

    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    /**
     * @brief Get a view representation of the matrix's data.
     * @return MatrixView for easy access to matrix elements.
     */
    [[nodiscard]] SparsityView<IndexType> view() const
    {
        return SparsityView<IndexType>(colIdxs_.view(), rowOffs_.view());
    }

    KOKKOS_INLINE_FUNCTION IndexType rowOffs(localIdx celli) const { return rowOffsV_[celli]; }

private:

    Vector<IndexType> rowOffs_; //! rowOffs map from row to start index in values

    Vector<IndexType> colIdxs_; //!

    View<IndexType> rowOffsV_;

    View<IndexType> colIdxsV_;
};


} // namespace NeoN::la
