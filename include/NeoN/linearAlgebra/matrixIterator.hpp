// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{


/* @class MatrixIterator
 * @brief class storing the mapping between mesh faces and target matrix sparsity pattern
 *
 * Based on given computational mesh this class stores a mapping for a consistent iteration
 * procedure for matrices which share the same sparsity pattern.
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 *
 */
template<typename ValueType, typename IndexType = localIdx, typename MeshType = UnstructuredMesh>
class MatrixIterator
{

    // const MeshType& mesh;

    // // the corresponding values
    // std::shared_ptr<Vector<ValueType>> values_;

    // NOTE The following data member store a simple mapping from face ids to offsets in the
    // corresponding rows I.e. Assume the following row  [ . 18 . 20 d 18 . 20 . ] this yields
    // ownerOffset[18] = 0 ownerOffset[20] = 1 and neighbourOffset[18] = 4 neighbourOffset[20] = 5
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

public:

    /* @brief create an SparsityPattern from existing mesh */
    MatrixIterator(
        Array<uint8_t> ownerOffset,
        Array<uint8_t> neighbourOffset,
        Array<uint8_t> diagOffset,
        std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern,
        std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsityPattern
    );

    std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern() const { return sp_; }

    std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsityPattern() const
    {
        return bsp_;
    }

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
    /* @brief given celli the function returns index of the diagonal element  */
    KOKKOS_INLINE_FUNCTION localIdx diagIdx(localIdx celli) const
    {
        return sp_->rowOffs(celli) + diagOffsetV_[celli];
    }

    /* @brief given celli the function returns index of the diagonal element  */
    KOKKOS_INLINE_FUNCTION localIdx upperIdx(localIdx celli, localIdx faceIdx) const
    {
        return sp_->rowOffs(celli) + neighbourOffsetV_[faceIdx];
    }

    /* @brief given celli the function returns index of the diagonal element  */
    KOKKOS_INLINE_FUNCTION localIdx lowerIdx(localIdx celli, localIdx faceIdx) const
    {
        return sp_->rowOffs(celli) + ownerOffsetV_[faceIdx];
    }
};

template<typename ValueType, typename IndexType>
MatrixIterator<ValueType, IndexType>
createSparsityPatternMatrixIterator(const UnstructuredMesh& mesh);

}
