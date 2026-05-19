// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

namespace NeoN::la
{

// ---------------------------------------------------------------------------
// FaceToMatrixAddress
// ---------------------------------------------------------------------------

void FaceToMatrixAddress::validate() const
{
    NF_ASSERT(ownerOffset_.exec() == neighbourOffset_.exec(), "Executors are not the same");
    NF_ASSERT(ownerOffset_.exec() == diagOffset_.exec(), "Executors are not the same");
    NF_ASSERT(
        ownerOffset_.size() == neighbourOffset_.size(),
        "ownerOffset and neighbourOffset must have the same size"
    );
}

FaceToMatrixAddress::FaceToMatrixAddress(
    Array<uint8_t> ownerOffset,
    Array<uint8_t> neighbourOffset,
    Array<uint8_t> diagOffset,
    std::shared_ptr<const CsrSparsityPattern<localIdx>> sparsityPattern,
    std::shared_ptr<const CooSparsityPattern<localIdx>> boundarySparsityPattern,
    std::shared_ptr<const CooSparsityPattern<localIdx>> nonLocalSparsityPattern
)
    : ownerOffset_(std::move(ownerOffset)), neighbourOffset_(std::move(neighbourOffset)),
      diagOffset_(std::move(diagOffset)), ownerOffsetV_(ownerOffset_.view()),
      neighbourOffsetV_(neighbourOffset_.view()), diagOffsetV_(diagOffset_.view()),
      sp_(sparsityPattern), bsp_(boundarySparsityPattern), nonLocalSp_(nonLocalSparsityPattern),
      rowOffsV_(sp_->rowOffs().view())
{
    validate();
}

FaceToMatrixAddress::FaceToMatrixAddress(const FaceToMatrixAddress& mi)
    : ownerOffset_(mi.ownerOffset_), neighbourOffset_(mi.neighbourOffset_),
      diagOffset_(mi.diagOffset_), ownerOffsetV_(ownerOffset_.view()),
      neighbourOffsetV_(neighbourOffset_.view()), diagOffsetV_(diagOffset_.view()), sp_(mi.sp_),
      bsp_(mi.bsp_), nonLocalSp_(mi.nonLocalSp_), rowOffsV_(sp_->rowOffs().view())
{
    validate();
}

FaceToMatrixAddress FaceToMatrixAddress::copyToHost() const
{
    return FaceToMatrixAddress(
        ownerOffset_.copyToHost(),
        neighbourOffset_.copyToHost(),
        diagOffset_.copyToHost(),
        std::make_shared<const CsrSparsityPattern<localIdx>>(sp_->copyToHost()),
        std::make_shared<const CooSparsityPattern<localIdx>>(bsp_->copyToHost()),
        std::make_shared<const CooSparsityPattern<localIdx>>(nonLocalSp_->copyToHost())
    );
}

FaceToMatrixAddress FaceToMatrixAddress::copyToExecutor(Executor dstExec) const
{
    return FaceToMatrixAddress(
        ownerOffset_.copyToExecutor(dstExec),
        neighbourOffset_.copyToExecutor(dstExec),
        diagOffset_.copyToExecutor(dstExec),
        std::make_shared<const CsrSparsityPattern<localIdx>>(sp_->copyToExecutor(dstExec)),
        std::make_shared<const CooSparsityPattern<localIdx>>(bsp_->copyToExecutor(dstExec)),
        std::make_shared<const CooSparsityPattern<localIdx>>(nonLocalSp_->copyToExecutor(dstExec))
    );
}

FaceToMatrixView FaceToMatrixAddress::view() const
{
    return FaceToMatrixView {ownerOffsetV_, neighbourOffsetV_, diagOffsetV_, rowOffsV_};
}

FaceToMatrixView FaceToMatrixAddress::view(View<const localIdx> rowOffsView) const
{
    return FaceToMatrixView {ownerOffsetV_, neighbourOffsetV_, diagOffsetV_, rowOffsView};
}

const NeoN::Array<uint8_t>& FaceToMatrixAddress::ownerOffset() const { return ownerOffset_; }

const NeoN::Array<uint8_t>& FaceToMatrixAddress::neighbourOffset() const
{
    return neighbourOffset_;
}

const NeoN::Array<uint8_t>& FaceToMatrixAddress::diagOffset() const { return diagOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::ownerOffset() { return ownerOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::neighbourOffset() { return neighbourOffset_; }

NeoN::Array<uint8_t>& FaceToMatrixAddress::diagOffset() { return diagOffset_; }

// ---------------------------------------------------------------------------
// computeRowToDiagonalMap
// ---------------------------------------------------------------------------

Vector<localIdx> computeRowToDiagonalMap(
    const Vector<localIdx>& rows, std::shared_ptr<const FaceToMatrixAddress> matIt
)
{
    const auto exec = rows.exec();
    const localIdx n = rows.size();
    Vector<localIdx> result(exec, n, 0);
    const auto rowsV = rows.view();
    auto resultV = result.view();
    const auto ma = matIt->view();
    parallelFor(
        exec,
        {0, n},
        KOKKOS_LAMBDA(const localIdx i) { resultV[i] = ma.diagIdx(rowsV[i]); },
        "computeRowToDiagonalMap"
    );
    return result;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

template<typename IndexType>
void setBoundarySparsityPatternImpl(
    const UnstructuredMesh& mesh,
    const Array<uint8_t>& diagOffs,
    Vector<IndexType>& rowIdx,
    Vector<IndexType>& colIdx
)
{
    const auto exec = mesh.exec();
    const auto diagOffsV = diagOffs.view();
    const auto faceCellsV = mesh.boundaryMesh().faceCells().view();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    NF_ASSERT(nBoundaryFaces == rowIdx.size(), "Inconsistent size");
    NF_ASSERT(nBoundaryFaces == colIdx.size(), "Inconsistent size");

    auto rowIdxV = rowIdx.view();
    auto colIdxV = colIdx.view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            localIdx celli = faceCellsV[bfacei];
            colIdxV[bfacei] = celli + diagOffsV[celli];
            rowIdxV[bfacei] = celli;
        },
        "setSparsityPatternFaceToMatrixAddress::setBoundarySparsity"
    );
}

template<typename IndexType>
void setProcBoundarySparsityPattern(
    const UnstructuredMesh& mesh,
    const Array<uint8_t>& diagOffs,
    Vector<IndexType>& rowIdx,
    Vector<IndexType>& colIdx
)
{
    const auto exec = mesh.exec();
    const auto faceCellsV = mesh.boundaryMesh().faceCells().view();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    auto rowIdxV = rowIdx.view();

    NF_ASSERT(nBoundaryFaces + nProcBoundaryFaces == faceCellsV.size(), "Inconsistent size");
    NF_ASSERT(nProcBoundaryFaces == rowIdx.size(), "Inconsistent size");
    NF_ASSERT(nProcBoundaryFaces == colIdx.size(), "Inconsistent size");

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            rowIdxV[bfacei] = faceCellsV[bfacei + nBoundaryFaces];
        },
        "setSparsityPatternFaceToMatrixAddress::setProcBoundarySparsity"
    );
}

template<typename IndexType>
void setSparsityPatternFaceToMatrixAddressSerial(
    const UnstructuredMesh& mesh,
    Array<uint8_t>& diagOffs,
    Array<uint8_t>& ownOffs,
    Array<uint8_t>& neiOffs,
    Vector<IndexType>& rowOffs,
    Vector<IndexType>& colIdx
)
{
    const auto nInternalFaces = mesh.nInternalFaces();
    auto nCells = mesh.nCells();

    // Start with one to include the diagonal
    auto nFacesPerCellH = Vector<localIdx>(SerialExecutor {}, nCells, 1);
    auto [neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH] =
        copyToHosts(neiOffs, ownOffs, diagOffs, mesh.faceOwner(), mesh.faceNeighbour());

    auto [nFacesPerCellHV, neiOffsetHV, ownOffsetHV, diagOffsetHV, faceOwnHV, faceNeiHV] =
        views(nFacesPerCellH, neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH);

    // Accumulate number of non-zeros per row
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto own = faceOwnHV[facei];
            auto nei = faceNeiHV[facei];
            Kokkos::atomic_inc(&nFacesPerCellHV[own]);
            Kokkos::atomic_inc(&nFacesPerCellHV[nei]);
        },
        "setSparsityPatternFaceToMatrixAddress::accumulateNonZeros"
    );

    auto rowOffsH = rowOffs.copyToHost();
    auto rowOffsHV = rowOffsH.view();
    segmentsFromIntervals(nFacesPerCellH, rowOffsH);
    auto colIdxH = colIdx.copyToHost();
    auto colIdxHV = colIdxH.view();
    fill(nFacesPerCellH, 0); // Reset nFacesPerCell

    // Compute lower triangular part
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellHV[nei], 1);
            neiOffsetHV[facei] = static_cast<uint8_t>(segIdxNei);
            auto startSegNei = rowOffsHV[nei];
            Kokkos::atomic_store(&colIdxHV[startSegNei + segIdxNei], own);
        },
        "setSparsityPatternFaceToMatrixAddress::computeLowerTriangular"
    );

    map(
        nFacesPerCellH,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            diagOffsetHV[celli] = static_cast<uint8_t>(nFaces);
            colIdxHV[rowOffsHV[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // Compute upper triangular part
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellHV[own], 1));
            ownOffsetHV[facei] = segIdxOwn;
            auto startSegOwn = rowOffsHV[own];
            Kokkos::atomic_store(&colIdxHV[startSegOwn + segIdxOwn], nei);
        },
        "setSparsityPatternFaceToMatrixAddress::computeUpperTriangular"
    );

    // Copy results back to device
    const auto exec = mesh.exec();
    ownOffs = ownOffsetH.copyToExecutor(exec);
    neiOffs = neiOffsetH.copyToExecutor(exec);
    diagOffs = diagOffsetH.copyToExecutor(exec);
    colIdx = colIdxH.copyToExecutor(exec);
    rowOffs = rowOffsH.copyToExecutor(exec);
}

// ---------------------------------------------------------------------------
// createSparsityPatternFaceToMatrixAddress
// ---------------------------------------------------------------------------

template<typename IndexType>
std::pair<std::shared_ptr<const FaceToMatrixAddress>, CommunicationPattern>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    const auto commPattern = computeCommunicationPattern(mesh);

    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<IndexType> rowOffs(exec, nCells + 1, 0);
    Vector<IndexType> colIdx(exec, nCells + 2 * nInternalFaces, 0);

    setSparsityPatternFaceToMatrixAddressSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);

    auto sp = std::make_shared<const CsrSparsityPattern<IndexType>>(
        std::move(colIdx), std::move(rowOffs), Dimensions {nCells, nCells}
    );

    // Physical boundary sparsity pattern (COO)
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    Vector<IndexType> bRowIdx(exec, nBoundaryFaces, 0);
    Vector<IndexType> bColIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, diagOffs, bRowIdx, bColIdx);
    auto bsp = std::make_shared<const CooSparsityPattern<IndexType>>(
        std::move(bColIdx), std::move(bRowIdx), Dimensions {nCells, nCells}
    );

    // Processor-boundary (nonLocal) sparsity pattern (COO)
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    Vector<IndexType> procRowIdx(exec, nProcBoundaryFaces, 0);
    Vector<IndexType> procColIdx(exec, nProcBoundaryFaces, 0);
    setProcBoundarySparsityPattern(mesh, diagOffs, procRowIdx, procColIdx);
    auto nonLocalSp = std::make_shared<const CooSparsityPattern<IndexType>>(
        std::move(procColIdx), std::move(procRowIdx), Dimensions {nCells, nCells}
    );

    return {
        std::make_shared<const FaceToMatrixAddress>(
            ownOffs, neiOffs, diagOffs, sp, bsp, nonLocalSp
        ),
        commPattern
    };
}

// ---------------------------------------------------------------------------
// createBoundarySparsityPattern specializations
// ---------------------------------------------------------------------------

template<>
std::shared_ptr<const CooSparsityPattern<localIdx>>
createBoundarySparsityPattern<CooSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& ftma
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    Vector<localIdx> rowIdx(exec, nBoundaryFaces, 0);
    Vector<localIdx> colIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, ftma.diagOffset(), rowIdx, colIdx);
    return std::make_shared<const CooSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowIdx), Dimensions {mesh.nCells(), mesh.nCells()}
    );
}

template<>
std::shared_ptr<const CsrSparsityPattern<localIdx>>
createBoundarySparsityPattern<CsrSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress& ftma
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    Vector<localIdx> rowIdx(exec, nBoundaryFaces, 0);
    Vector<localIdx> colIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, ftma.diagOffset(), rowIdx, colIdx);
    auto rowPtrs = rowsToRowOffs(rowIdx);
    return std::make_shared<const CsrSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowPtrs), Dimensions {mesh.nCells(), nBoundaryFaces}
    );
}

template std::pair<std::shared_ptr<const FaceToMatrixAddress>, CommunicationPattern>
createSparsityPatternFaceToMatrixAddress<localIdx>(const UnstructuredMesh&);

} // namespace NeoN::la
