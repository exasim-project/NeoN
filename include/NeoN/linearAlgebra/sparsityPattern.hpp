// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{

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

    KOKKOS_INLINE_FUNCTION IndexType rowOffs(localIdx celli) const { return rowOffsV_[celli]; }

private:

    Vector<IndexType> rowOffs_; //! rowOffs map from row to start index in values

    Vector<IndexType> colIdxs_; //!

    View<IndexType> rowOffsV_;

    View<IndexType> colIdxsV_;
};


} // namespace NeoN::la
