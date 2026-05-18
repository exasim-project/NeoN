// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/sparsityView.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{


/** @class CooSparsityPattern
 * @brief row and column index representation of a mesh
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 */
template<typename IndexType>
class CooSparsityPattern
{

    void validate() const;

public:

    using SparsityIndexType = IndexType;

    /** @brief create an "empty" SparsityPattern with a given size  */
    CooSparsityPattern(const CooSparsityPattern& sp);

    CooSparsityPattern(Vector<IndexType>&& colIdx, Vector<IndexType>&& rowIdxs, Dimensions dim);

    [[nodiscard]] CooSparsityPattern copyToHost() const
    {
        return CooSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(SerialExecutor()),
            rowIdxs_.copyToExecutor(SerialExecutor()),
            dimensions_
        );
    }

    [[nodiscard]] CooSparsityPattern copyToExecutor(Executor dstExec) const
    {
        return CooSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(dstExec), rowIdxs_.copyToExecutor(dstExec), dimensions_
        );
    }


    ~CooSparsityPattern() = default;

    /**@brief getter for executor */
    const Executor& exec() const { return rowOffs_.exec(); }

    /**@brief const getter for colIdxs */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    /**@brief getter for colIdxs */
    [[nodiscard]] Vector<IndexType>& colIdxs() { return colIdxs_; };

    /**@brief returns the COO per-entry row indices (one per nnz) */
    [[nodiscard]] const Vector<IndexType>& rowIdxs() const { return rowIdxs_; };

    /**@brief returns the non-const COO per-entry row indices (one per nnz) */
    [[nodiscard]] Vector<IndexType>& rowIdxs() { return rowIdxs_; };

    /**@brief returns the COO per-entry row indices (one per nnz) */
    [[nodiscard]] const Vector<IndexType>& rowOffs() const { return rowOffs_; };

    /**@brief returns the non-const COO per-entry row indices (one per nnz) */
    [[nodiscard]] Vector<IndexType>& rowOffs() { return rowOffs_; };

    [[nodiscard]] localIdx rows() const { return dimensions_.rows; };

    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    [[nodiscard]] Dimensions dimension() const { return dimensions_; };

    /**
     * @brief Get a view representation of the matrix's data.
     * @return MatrixView for easy access to matrix elements.
     */
    [[nodiscard]] SparsityView<IndexType> view() const
    {
        return SparsityView<IndexType>(colIdxs_.view(), rowIdxs_.view());
    }

private:

    Dimensions dimensions_;

    Vector<IndexType> rowIdxs_; //! rowIdx

    Vector<IndexType> colIdxs_; //!

    // NOTE for now the rowOffsets are kept since it is required for face based
    // matrix assembly.
    Vector<IndexType> rowOffs_; //! rowOffs map from row to start index in values
};

} // namespace NeoN::la
