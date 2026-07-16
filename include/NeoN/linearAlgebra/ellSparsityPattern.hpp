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

    /**
     * @brief construct from a fully-built padded, column-major colIdx array.
     * @param colIdx per-slot column indices, sorted ascending within each row's slots,
     * with padding (EllSparsityView<IndexType>::invalidIndex()) trailing. Not verified.
     * @param logicalNnz count of non-padding entries in colIdx. Caller-supplied and
     * trusted, not recomputed from colIdx -- only checked against storage size.
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

    /*@brief const-only getter for colIdxs -- ELL patterns are immutable after construction
     * so nnz() can't desync from the stored columns */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] localIdx rows() const { return dimensions_.rows; };

    /**
     * @brief size of the (padded) storage backing this pattern, i.e. the size
     * `values_` of a `Matrix` built on top of this pattern must have.
     * @note unlike CSR, this includes padding entries and is therefore not the
     * count of logical/non-zero matrix entries -- see nnz().
     */
    [[nodiscard]] localIdx storageSize() const { return colIdxs_.size(); };

    /*@brief true count of logical (non-padding) nonzero matrix entries */
    [[nodiscard]] localIdx nnz() const { return logicalNnz_; };

    /*@brief number of column-index slots stored per row, including padding */
    [[nodiscard]] localIdx numStoredElementsPerRow() const { return numStoredElementsPerRow_; };

    /*@brief stride between successive slots of the column-major-stored ELL arrays */
    [[nodiscard]] localIdx stride() const { return stride_; };

    [[nodiscard]] Dimensions dimension() const { return dimensions_; };

    using ViewType = EllSparsityView<IndexType>;

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

    localIdx logicalNnz_; //! true count of non-padding nonzero entries, <= storageSize()
};

} // namespace NeoN::la
