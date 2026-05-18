// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityView.hpp"

namespace NeoN::la
{

/** @class CsrSparsityPattern
 * @brief row and column index representation of a mesh
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 */
template<typename IndexType>
class CsrSparsityPattern
{

    void validate() const;

public:

    using SparsityIndexType = IndexType;

    /* @brief create an "empty" SparsityPattern with a given size  */
    CsrSparsityPattern(const CsrSparsityPattern& sp);

    CsrSparsityPattern(Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs, Dimensions dim);

    [[nodiscard]] CsrSparsityPattern copyToHost() const
    {
        return CsrSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(SerialExecutor()),
            rowOffs_.copyToExecutor(SerialExecutor()),
            dimensions_
        );
    }

    [[nodiscard]] CsrSparsityPattern copyToExecutor(Executor dstExec) const
    {
        return CsrSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(dstExec), rowOffs_.copyToExecutor(dstExec), dimensions_
        );
    }


    ~CsrSparsityPattern() = default;

    /*@brief getter for executor */
    const Executor& exec() const { return rowOffs_.exec(); }

    /*@brief getter for colIdxs */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] Vector<IndexType>& colIdxs() { return colIdxs_; };

    /*@brief getter for rowOffs */
    [[nodiscard]] const Vector<IndexType>& rowOffs() const { return rowOffs_; };

    /*@brief getter for non-const rowOffs */
    [[nodiscard]] Vector<IndexType>& rowOffs() { return rowOffs_; };

    [[nodiscard]] localIdx rows() const { return rowOffs_.size() - 1; };

    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    [[nodiscard]] Dimensions dimension() const { return dimensions_; };
    /**
     * @brief Get a view representation of the matrix's data.
     * @return MatrixView for easy access to matrix elements.
     */
    [[nodiscard]] SparsityView<IndexType> view() const
    {
        return SparsityView<IndexType>(colIdxs_.view(), rowOffs_.view());
    }

private:

    Dimensions dimensions_;

    Vector<IndexType> rowOffs_; //! rowOffs map from row to start index in values

    Vector<IndexType> colIdxs_; //!
};

} // namespace NeoN::la
