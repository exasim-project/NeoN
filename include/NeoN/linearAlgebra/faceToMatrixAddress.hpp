// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/core/copyTo.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
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
class FaceToMatrixAddress : public NeoN::SupportsCopyTo<FaceToMatrixAddress>
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


    FaceToMatrixAddress copyToExecutor(Executor dstExec) const;

    /**
     * @brief Get a view representation of the matrix's data.
     * @return FaceToMatrixView for easy access to matrix elements.
     */
    [[nodiscard]] FaceToMatrixView view(View<const localIdx> rowOffsView) const;

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
std::pair<std::shared_ptr<const SparsityType>, std::shared_ptr<const FaceToMatrixAddress>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh);

/* @brief Creates the boundary sparsity pattern from a mesh and an existing
 * FaceToMatrixAddress (which provides the diagonal offsets needed to compute it).
 *
 * @tparam SparsityType - The full sparsity pattern type to create, e.g.
 *         CooSparsityPattern<localIdx> or CsrSparsityPattern<localIdx>
 */
template<typename SparsityType>
std::shared_ptr<const SparsityType> createBoundarySparsityPattern(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& faceToMatrixAddress
);

template<typename SparsityType>
std::shared_ptr<const SparsityType> createOffDiagonalSparsityPattern(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& faceToMatrixAddress
);

}
