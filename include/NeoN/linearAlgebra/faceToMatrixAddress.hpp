// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/distributed/communicationPattern.hpp"

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

    // the common local sparsity pattern
    std::shared_ptr<const SparsityPattern<IndexType>> sp_;

    // the common non-local sparsity pattern
    std::shared_ptr<const CooSparsityPattern<IndexType>> nonLocalSp_;

    // the common boundary sparsity pattern
    std::shared_ptr<const CooSparsityPattern<IndexType>> bsp_;

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
        std::shared_ptr<const CooSparsityPattern<IndexType>> nonLocalSparsityPattern,
        std::shared_ptr<const CooSparsityPattern<IndexType>> boundarySparsityPattern
    );

    /* @brief copy constructor */
    FaceToMatrixAddress(const FaceToMatrixAddress& mi);

    FaceToMatrixAddress copyToHost() const;

    std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern() const { return sp_; }

    std::shared_ptr<const CooSparsityPattern<IndexType>> nonLocalSparsityPattern() const
    {
        return nonLocalSp_;
    }

    std::shared_ptr<const CooSparsityPattern<IndexType>> boundarySparsityPattern() const
    {
        return bsp_;
    }

    const Executor& exec() const { return sp_->exec(); }

    /*@brief return the number of rows in local matrix */
    localIdx localRows() const { return sp_->rows(); };

    /*@brief return the number of non-zeros in local matrix */
    localIdx localNonZeros() const { return sp_->nnz(); };

    /*@brief return the number of non-zeros in local matrix */
    localIdx nonLocalNonZeros() const { return nonLocalSp_->nnz(); };

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

template<typename IndexType>
std::shared_ptr<const FaceToMatrixAddress<IndexType>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh);

template<typename IndexType>
std::shared_ptr<const FaceToMatrixAddress<IndexType>> createSparsityPatternFaceToMatrixAddressDist(
    const UnstructuredMesh& mesh, CommunicationPattern& commPattern
);

/**@brief given a set of rows this function computes the corresponding value offsets */
inline Vector<localIdx> computeRowToDiagonalMap(
    const Vector<localIdx>& rows, std::shared_ptr<const FaceToMatrixAddress<>> matIt
)
{

    const auto exec = rows.exec();
    Vector<localIdx> ret(exec, rows.size());
    auto retV = ret.view();
    const auto rowsV = rows.view();
    const auto [rowOffs, diagOffs] =
        views(matIt->sparsityPattern()->rowOffs(), matIt->diagOffset());

    parallelFor(
        exec,
        {0, ret.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            localIdx cell = rowsV[i];
            auto rowStart = rowOffs[cell];
            retV[i] = rowStart + diagOffs[cell];
        },
        "computeRowToDiagonalMap"
    );

    return ret;
}


}
