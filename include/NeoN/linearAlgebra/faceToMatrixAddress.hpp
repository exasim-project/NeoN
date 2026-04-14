// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{


/* @class FaceToMatrixAddress
 * @brief Stores the mapping between mesh faces and matrix row offsets.
 *
 * Based on a given computational mesh this class stores a mapping for a consistent iteration
 * procedure for matrices which share the same sparsity pattern.
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of face-to-matrix offset mappings. The sparsity pattern itself is owned by
 * the Matrix, not by this class; this class only borrows a view of the row offsets.
 *
 */
template<typename IndexType = localIdx, typename MeshType = UnstructuredMesh>
class FaceToMatrixAddress
{

    // clang-format off
    // NOTE The following data members store a simple mapping from face ids to offsets within
    // the corresponding row of the CSR values array.
    //
    // For an internal face f with owner P and neighbour N (P < N by construction):
    //   ownerOffset[f]      = offset within row P for column N  → A[P, N]  (upper triangular)
    //   neighbourOffset[f]  = offset within row N for column P  → A[N, P]  (lower triangular)
    //
    // Example row layout for a cell with two lower entries and one upper entry:
    //   Row P: [ A[P,j0]  A[P,j1] | A[P,P] | A[P,N] ]
    //           lower (j<P)         diag      upper (N>P)
    //   ownerOffset[f]     = 3   (position of A[P,N] within row P)
    //   neighbourOffset[f] = 0   (position of A[N,P] within row N)
    // clang-format on
    Array<uint8_t>
        ownerOffset_; //! offset within the owner's row for the upper-triangular entry A[own, nei]

    Array<uint8_t> neighbourOffset_; //! offset within the neighbour's row for the lower-triangular
                                     //! entry A[nei, own]

    Array<uint8_t> diagOffset_; //! mapping from celli to diagonal element offset

    // corresponding views
    View<uint8_t> ownerOffsetV_;

    View<uint8_t> neighbourOffsetV_;

    View<uint8_t> diagOffsetV_;

    // row offsets borrowed from the owning Matrix's sparsity pattern
    View<const IndexType> rowOffsV_;

public:

    /* @brief constructor
     *
     * @param ownerOffset     face-to-lower-offset mapping
     * @param neighbourOffset face-to-upper-offset mapping
     * @param diagOffset      cell-to-diagonal-offset mapping
     * @param rowOffsView     view of row offsets from the owning Matrix's CsrSparsityPattern
     */
    FaceToMatrixAddress(
        Array<uint8_t> ownerOffset,
        Array<uint8_t> neighbourOffset,
        Array<uint8_t> diagOffset,
        View<const IndexType> rowOffsView
    );

    /* @brief copy constructor */
    FaceToMatrixAddress(const FaceToMatrixAddress& mi);

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
    /* @brief Returns the flat values-array index of the diagonal entry for cell celli.
     *  diagIdx(celli) = rowOffs[celli] + diagOffset[celli]
     */
    KOKKOS_INLINE_FUNCTION localIdx diagIdx(localIdx celli) const
    {
        return rowOffsV_[celli] + diagOffsetV_[celli];
    }

    /* @brief Returns the flat values-array index of the upper-triangular entry A[own, nei].
     *
     *  By construction own < nei for every internal face, so the column index nei is greater
     *  than the row index own — this entry lies in the upper triangle.
     *
     *  upperIdx(own, f) = rowOffs[own] + ownerOffset[f]
     */
    KOKKOS_INLINE_FUNCTION localIdx upperIdx(localIdx own, localIdx faceIdx) const
    {
        return rowOffsV_[own] + ownerOffsetV_[faceIdx];
    }

    /* @brief Returns the flat values-array index of the lower-triangular entry A[nei, own].
     *
     *  By construction nei > own for every internal face, so the column index own is less
     *  than the row index nei — this entry lies in the lower triangle.
     *
     *  lowerIdx(nei, f) = rowOffs[nei] + neighbourOffset[f]
     */
    KOKKOS_INLINE_FUNCTION localIdx lowerIdx(localIdx nei, localIdx faceIdx) const
    {
        return rowOffsV_[nei] + neighbourOffsetV_[faceIdx];
    }
};

/* @brief Creates the sparsity pattern and corresponding FaceToMatrixAddress from a mesh.
 *
 * The two are returned together because FaceToMatrixAddress borrows the row-offsets
 * view from the sparsity pattern. The boundary sparsity is created separately
 * via createBoundarySparsityPattern.
 *
 * @tparam SparsityType - The full sparsity pattern type to create, e.g.
 *         CsrSparsityPattern<localIdx> or CooSparsityPattern<localIdx>
 */
template<typename SparsityType>
std::pair<
    std::shared_ptr<const SparsityType>,
    std::shared_ptr<const FaceToMatrixAddress<typename SparsityType::SparsityIndexType>>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh);

/* @brief Creates the boundary sparsity pattern from a mesh and an existing
 * FaceToMatrixAddress (which provides the diagonal offsets needed to compute it).
 *
 * @tparam SparsityType - The full sparsity pattern type to create, e.g.
 *         CooSparsityPattern<localIdx> or CsrSparsityPattern<localIdx>
 */
template<typename SparsityType>
std::shared_ptr<const SparsityType> createBoundarySparsityPattern(
    const UnstructuredMesh& mesh,
    const FaceToMatrixAddress<typename SparsityType::SparsityIndexType>& faceToMatrixAddress
);

}
