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
 * @brief fixed-width, padded column-index representation of a matrix (ELLPACK format)
 *
 * Every row reserves `numStoredElementsPerRow()` column-index slots, the width of
 * the widest row in the matrix. Rows with fewer nonzeros are padded with
 * `EllSparsityView<IndexType>::invalidIndex()`. Slots are stored column-major,
 * i.e. slot `s` of row `i` is stored at `i + stride() * s`, so that a
 * one-thread-per-row kernel reads/writes memory in a coalesced fashion when
 * iterating over slots.
 */
template<typename IndexType>
class EllSparsityPattern : public NeoN::SupportsCopyTo<EllSparsityPattern<IndexType>>
{

    void validate() const;

public:

    using SparsityIndexType = IndexType;

    /* @brief create a copy of a given EllSparsityPattern */
    EllSparsityPattern(const EllSparsityPattern& sp);

    EllSparsityPattern(
        Vector<IndexType>&& colIdx,
        Dimensions dim,
        localIdx numStoredElementsPerRow,
        localIdx stride
    );

    [[nodiscard]] EllSparsityPattern copyToExecutor(Executor dstExec) const override
    {
        return EllSparsityPattern<IndexType>(
            colIdxs_.copyToExecutor(dstExec), dimensions_, numStoredElementsPerRow_, stride_
        );
    }

    ~EllSparsityPattern() = default;

    /*@brief getter for executor */
    const Executor& exec() const { return colIdxs_.exec(); }

    /*@brief getter for colIdxs */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] Vector<IndexType>& colIdxs() { return colIdxs_; };

    [[nodiscard]] localIdx rows() const { return dimensions_.rows; };

    /**
     * @brief size of the (padded) storage backing this pattern, i.e. the size
     * `values_` of a `Matrix` built on top of this pattern must have.
     * @note unlike CSR, this includes padding entries and is therefore not the
     * count of logical/non-zero matrix entries.
     */
    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    /*@brief number of column-index slots stored per row, including padding */
    [[nodiscard]] localIdx numStoredElementsPerRow() const { return numStoredElementsPerRow_; };

    /*@brief stride between successive slots of the column-major-stored ELL arrays */
    [[nodiscard]] localIdx stride() const { return stride_; };

    [[nodiscard]] Dimensions dimension() const { return dimensions_; };

    /**
     * @brief Get a view representation of the matrix's data.
     * @return EllSparsityView for easy access to matrix elements.
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

    Vector<IndexType> colIdxs_; //! padded, column-major column indices,
                                //! size stride_ * numStoredElementsPerRow_

    localIdx numStoredElementsPerRow_; //! width of the widest row, i.e. slots stored per row

    localIdx stride_; //! distance between slot s and slot s+1 of the same row
};

} // namespace NeoN::la
