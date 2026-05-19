// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/distributed/communicationPattern.hpp"

namespace NeoN::la
{

/**
 * @struct FaceToMatrixView
 * @brief A device-accessible view into the face-to-matrix address mapping.
 *
 * Holds lightweight Kokkos views (no ownership) of the offset arrays and the
 * row-offset array borrowed from the associated CSR sparsity pattern.
 * All members are safe to use inside Kokkos kernels.
 */
struct FaceToMatrixView
{
    // TODO check performance
    /** @brief flat values-array index of the diagonal entry for cell celli */
    KOKKOS_INLINE_FUNCTION localIdx diagIdx(localIdx celli) const
    {
        return rowOffs[celli] + diagOffset[celli];
    }

    /** @brief flat values-array index of the upper-triangular entry A[own, nei] */
    KOKKOS_INLINE_FUNCTION localIdx upperIdx(localIdx own, localIdx faceIdx) const
    {
        return rowOffs[own] + ownerOffset[faceIdx];
    }

    /** @brief flat values-array index of the lower-triangular entry A[nei, own] */
    KOKKOS_INLINE_FUNCTION localIdx lowerIdx(localIdx nei, localIdx faceIdx) const
    {
        return rowOffs[nei] + neighbourOffset[faceIdx];
    }

    View<const uint8_t> ownerOffset;
    View<const uint8_t> neighbourOffset;
    View<const uint8_t> diagOffset;
    View<const localIdx> rowOffs;
};

/* @class FaceToMatrixAddress
 * @brief Stores the mapping between mesh faces and target matrix sparsity pattern,
 *        together with the three sparsity patterns (local CSR, boundary COO,
 *        nonLocal/processor-boundary COO).
 *
 * Based on a given computational mesh this class stores a mapping for a consistent
 * iteration procedure for matrices which share the same sparsity pattern.
 *
 * This class implements the finite-volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh.
 *
 * For an internal face f with owner P and neighbour N (P < N by construction):
 *   ownerOffset[f]      = offset within row P for column N  → A[P, N]  (upper triangular)
 *   neighbourOffset[f]  = offset within row N for column P  → A[N, P]  (lower triangular)
 */
class FaceToMatrixAddress
{
    Array<uint8_t> ownerOffset_;
    Array<uint8_t> neighbourOffset_;
    Array<uint8_t> diagOffset_;

    View<const uint8_t> ownerOffsetV_;
    View<const uint8_t> neighbourOffsetV_;
    View<const uint8_t> diagOffsetV_;
    View<const localIdx> rowOffsV_;

    std::shared_ptr<const CsrSparsityPattern<localIdx>> sp_;
    std::shared_ptr<const CooSparsityPattern<localIdx>> bsp_;
    std::shared_ptr<const CooSparsityPattern<localIdx>> nonLocalSp_;

    void validate() const;

public:

    FaceToMatrixAddress(
        Array<uint8_t> ownerOffset,
        Array<uint8_t> neighbourOffset,
        Array<uint8_t> diagOffset,
        std::shared_ptr<const CsrSparsityPattern<localIdx>> sparsityPattern,
        std::shared_ptr<const CooSparsityPattern<localIdx>> boundarySparsityPattern,
        std::shared_ptr<const CooSparsityPattern<localIdx>> nonLocalSparsityPattern
    );

    FaceToMatrixAddress(const FaceToMatrixAddress& mi);

    [[nodiscard]] FaceToMatrixAddress copyToHost() const;
    [[nodiscard]] FaceToMatrixAddress copyToExecutor(Executor dstExec) const;

    /** @brief Build a FaceToMatrixView using the internally stored row offsets. */
    [[nodiscard]] FaceToMatrixView view() const;

    /** @brief Build a FaceToMatrixView using an externally provided row-offsets view. */
    [[nodiscard]] FaceToMatrixView view(View<const localIdx> rowOffsView) const;

    [[nodiscard]] const Executor& exec() const { return sp_->exec(); }

    [[nodiscard]] localIdx localRows() const { return sp_->rows(); }

    [[nodiscard]] localIdx localNonZeros() const { return sp_->nnz(); }

    [[nodiscard]] localIdx boundaryRows() const { return bsp_->rows(); }

    [[nodiscard]] localIdx boundaryNonZeros() const { return bsp_->nnz(); }

    [[nodiscard]] localIdx nonLocalRows() const { return nonLocalSp_->rows(); }

    [[nodiscard]] localIdx nonLocalNonZeros() const { return nonLocalSp_->nnz(); }

    [[nodiscard]] std::shared_ptr<const CsrSparsityPattern<localIdx>> sparsityPattern() const
    {
        return sp_;
    }

    [[nodiscard]] std::shared_ptr<const CooSparsityPattern<localIdx>>
    boundarySparsityPattern() const
    {
        return bsp_;
    }

    [[nodiscard]] std::shared_ptr<const CooSparsityPattern<localIdx>>
    nonLocalSparsityPattern() const
    {
        return nonLocalSp_;
    }

    [[nodiscard]] const Array<uint8_t>& ownerOffset() const;
    [[nodiscard]] const Array<uint8_t>& neighbourOffset() const;
    [[nodiscard]] const Array<uint8_t>& diagOffset() const;
    [[nodiscard]] Array<uint8_t>& ownerOffset();
    [[nodiscard]] Array<uint8_t>& neighbourOffset();
    [[nodiscard]] Array<uint8_t>& diagOffset();

    KOKKOS_INLINE_FUNCTION localIdx diagIdx(localIdx celli) const
    {
        return rowOffsV_[celli] + diagOffsetV_[celli];
    }

    KOKKOS_INLINE_FUNCTION localIdx upperIdx(localIdx own, localIdx faceIdx) const
    {
        return rowOffsV_[own] + ownerOffsetV_[faceIdx];
    }

    KOKKOS_INLINE_FUNCTION localIdx lowerIdx(localIdx nei, localIdx faceIdx) const
    {
        return rowOffsV_[nei] + neighbourOffsetV_[faceIdx];
    }
};

/**@brief given a set of rows this function computes the corresponding diagonal value offsets */
Vector<localIdx> computeRowToDiagonalMap(
    const Vector<localIdx>& rows, std::shared_ptr<const FaceToMatrixAddress> matIt
);

/**
 * @brief Creates the local CSR, boundary COO and nonLocal COO sparsity patterns and
 *        packs them into a FaceToMatrixAddress.
 *
 * @tparam IndexType  Integer type for all sparsity indices (typically localIdx).
 * @return A pair containing:
 *         - shared_ptr<const FaceToMatrixAddress> holding all three sparsity patterns.
 *         - CommunicationPattern with MPI exchange metadata (empty for single-process runs).
 */
template<typename IndexType>
std::pair<std::shared_ptr<const FaceToMatrixAddress>, CommunicationPattern>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh);

/**
 * @brief Creates the boundary sparsity pattern from a mesh and an existing FaceToMatrixAddress.
 *
 * @tparam SparsityType  The sparsity pattern type to create, e.g. CooSparsityPattern<localIdx>.
 */
template<typename SparsityType>
std::shared_ptr<const SparsityType>
createBoundarySparsityPattern(const UnstructuredMesh& mesh, const FaceToMatrixAddress& ftma);

}
