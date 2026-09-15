// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/copyTo.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityView.hpp"

namespace NeoN::la
{

/** @class EllSparsityPattern
 * @brief fixed width, padded column index representation of a matrix (ELLPACK format)
 *
 * Every row stores numStoredElementsPerRow() column indices, the width of the widest
 * row, padding the shorter rows with invalidIndex(). The slots are stored column major,
 * slot s of row i at i + stride() * s, so that one thread per row accesses memory
 * coalesced.
 */
template<typename IndexType>
class EllSparsityPattern : public NeoN::SupportsCopyTo<EllSparsityPattern<IndexType>>
{

    void validate() const;

public:

    using SparsityIndexType = IndexType;

    /* @brief create a copy of a given EllSparsityPattern */
    EllSparsityPattern(const EllSparsityPattern& sp);

    /**
     * @brief construct from a padded, column major colIdx array
     * @param colIdx per slot column indices, sorted ascending per row with the padding
     * trailing
     * @param logicalNnz number of non padding entries, only checked against the storage size
     */
    EllSparsityPattern(
        Vector<IndexType>&& colIdx,
        Dimensions dim,
        localIdx numStoredElementsPerRow,
        localIdx stride,
        localIdx logicalNnz
    );

    [[nodiscard]] EllSparsityPattern copyToExecutor(Executor dstExec) const override
    {
        return EllSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(dstExec),
            dimensions_,
            numStoredElementsPerRow_,
            stride_,
            logicalNnz_
        );
    }

    ~EllSparsityPattern() = default;

    /*@brief getter for executor */
    const Executor& exec() const { return colIdxs_.exec(); }

    /*@brief const getter for colIdxs, non-const is omitted so nnz() cannot desync */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] localIdx rows() const { return dimensions_.rows; };

    /*@brief getter for the padded storage size, the size the values of a Matrix need */
    [[nodiscard]] localIdx storageSize() const { return colIdxs_.size(); };

    /*@brief getter for the number of non padding entries */
    [[nodiscard]] localIdx nnz() const { return logicalNnz_; };

    /*@brief getter for the number of slots stored per row, padding included */
    [[nodiscard]] localIdx numStoredElementsPerRow() const { return numStoredElementsPerRow_; };

    /*@brief getter for the stride between successive slots */
    [[nodiscard]] localIdx stride() const { return stride_; };

    [[nodiscard]] Dimensions dimension() const { return dimensions_; };

    using ViewType = EllSparsityView<IndexType>;

    /**
     * @brief Get a view representation of the sparsity pattern.
     * @return EllSparsityView for easy access to the stored entries.
     */
    [[nodiscard]] EllSparsityView<IndexType> view() const
    {
        return EllSparsityView<IndexType>(
            colIdxs_.view(),
            static_cast<IndexType>(numStoredElementsPerRow_),
            static_cast<IndexType>(stride_)
        );
    }

private:

    Dimensions dimensions_;

    Vector<IndexType> colIdxs_; //! padded column indices, size stride_ * numStoredElementsPerRow_

    localIdx numStoredElementsPerRow_; //! width of the widest row

    localIdx stride_; //! distance between slot s and slot s+1 of the same row

    localIdx logicalNnz_; //! number of non padding entries
};

} // namespace NeoN::la
