// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{


/* @class FaceToMatrixAddress
 * @brief class storing the mapping between mesh faces and target matrix sparsity pattern
 *
 * Based on given computational mesh this class stores a mapping for a consistent iteration
 * procedure for matrices which share the same sparsity pattern.
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 *
 */
template<typename IndexType = localIdx, typename MeshType = UnstructuredMesh>
class FaceToMatrixAddress
{

    // clang-format off
    // NOTE The following data member store a simple mapping from face ids to offsets in the
    // corresponding rows.
    // I.e. Assume the following row  [ . a_18 . a_20 d a_18 . a_20 . ]
    // where a_i corresponds to a value on a face i
    // this yields:
    // ownerOffset[18] = 0
    // ownerOffset[20] = 1
    // and
    // neighbourOffset[18] = 4
    // neighbourOffset[20] = 5
    // clang-format on
    Array<uint8_t> ownerOffset_; //! mapping from faceId to lower index in a row

    Array<uint8_t> neighbourOffset_; //! mapping from faceId to upper index in a row

    Array<uint8_t> diagOffset_; //! mapping from celli to diagonal element offset

    // corresponding views
    View<uint8_t> ownerOffsetV_;

    View<uint8_t> neighbourOffsetV_;

    View<uint8_t> diagOffsetV_;

    // the common sparsity pattern
    std::shared_ptr<const SparsityPattern<IndexType>> sp_;

    // the common boundary sparsity pattern
    std::shared_ptr<const SparsityPattern<IndexType>> bsp_;

    // start of a given row.
    View<const IndexType> rowOffsV_;

private:

    void validate() const;

public:

    /* @brief constructor setting members explicitly */
    FaceToMatrixAddress(
        Array<uint8_t> ownerOffset,
        Array<uint8_t> neighbourOffset,
        Array<uint8_t> diagOffset,
        std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern,
        std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsityPattern
    );

    /* @brief copy constructor */
    FaceToMatrixAddress(const FaceToMatrixAddress& mi);

    FaceToMatrixAddress copyToHost() const;

    std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern() const { return sp_; }

    std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsityPattern() const
    {
        return bsp_;
    }

    const Executor& exec() const { return sp_->exec(); }

    /*@brief return the number of rows in local matrix */
    localIdx localRows() const { return sp_->rows(); };

    /*@brief return the number of non-zeros in local matrix */
    localIdx localNonZeros() const { return sp_->nnz(); };

    /*@brief return the number of rows in boundary matrix */
    localIdx boundaryRows() const { return bsp_->rows(); };

    /*@brief return the number of non-zeros in boundary matrix */
    localIdx boundaryNonZeros() const { return bsp_->nnz(); };

    /*@brief getter for ownerOffset */
    const Array<uint8_t>& ownerOffset() const;

    /*@brief getter for neighbourOffset */
    const Array<uint8_t>& neighbourOffset() const;

    /*@brief getter for diagOffset */
    const Array<uint8_t>& diagOffset() const;

    /*@brief getter for ownerOffset */
    Array<uint8_t>& ownerOffset();

    /*@brief getter for neighbourOffset */
    Array<uint8_t>& neighbourOffset();

    /*@brief getter for diagOffset */
    Array<uint8_t>& diagOffset();

    // TODO check performance
    /* @brief given a cell ID, the function returns index of the diagonal element  */
    KOKKOS_INLINE_FUNCTION localIdx diagIdx(localIdx celli) const
    {
        return rowOffsV_[celli] + diagOffsetV_[celli];
    }

    /* @brief given a cell ID and a face ID, the function returns index of the element in the upper
     * triangular matrix */
    KOKKOS_INLINE_FUNCTION localIdx upperIdx(localIdx celli, localIdx faceIdx) const
    {
        return rowOffsV_[celli] + neighbourOffsetV_[faceIdx];
    }

    /* @brief given a cell ID and a face ID, the function returns index of the element in the lower
     * triangular matrix */
    KOKKOS_INLINE_FUNCTION localIdx lowerIdx(localIdx celli, localIdx faceIdx) const
    {
        return rowOffsV_[celli] + ownerOffsetV_[faceIdx];
    }
};

template<typename IndexType>
std::shared_ptr<const FaceToMatrixAddress<IndexType>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh);

}
