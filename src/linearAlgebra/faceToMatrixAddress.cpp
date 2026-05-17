// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

namespace NeoN::la
{

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::ownerOffset() const
{
    return ownerOffset_;
}

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::neighbourOffset() const
{
    return neighbourOffset_;
}

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::diagOffset() const
{
    return diagOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::ownerOffset()
{
    return ownerOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::neighbourOffset()
{
    return neighbourOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& FaceToMatrixAddress<IndexType, MeshType>::diagOffset()
{
    return diagOffset_;
}

template<typename IndexType, typename MeshType>
FaceToMatrixAddress<IndexType, MeshType>::FaceToMatrixAddress(
    Array<uint8_t> ownerOffset, Array<uint8_t> neighbourOffset, Array<uint8_t> diagOffset
)
    : ownerOffset_(ownerOffset), neighbourOffset_(neighbourOffset), diagOffset_(diagOffset)
{}

template<typename IndexType, typename MeshType>
FaceToMatrixAddress<IndexType, MeshType>::FaceToMatrixAddress(const FaceToMatrixAddress& mi)
    : ownerOffset_(mi.ownerOffset_), neighbourOffset_(mi.neighbourOffset_),
      diagOffset_(mi.diagOffset_)
{}

template class FaceToMatrixAddress<localIdx, UnstructuredMesh>;

// ---------------------------------------------------------------------------
// Internal helpers for building the sparsity data from a mesh
// ---------------------------------------------------------------------------

template<typename IndexType>
void setBoundarySparsityPatternImpl(
    const UnstructuredMesh& mesh,
    const Array<uint8_t>& diagOffs,
    Vector<IndexType>& bRowOffs,
    Vector<IndexType>& bColIdx
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    const auto diagOffsV = diagOffs.view();
    const auto faceCellsV = mesh.boundaryMesh().faceCells().view();
    auto bRowOffsV = bRowOffs.view();
    auto bColIdxV = bColIdx.view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            localIdx celli = faceCellsV[bfacei];
            bColIdxV[bfacei] = celli + diagOffsV[celli];
            bRowOffsV[bfacei] = celli;
        },
        "setSparsityPatternFaceToMatrixAddress::setBoundarySparsity"
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
    // TODO: currently the whole algorithm is performed in serial on the host
    // move it to executor
    const auto nInternalFaces = mesh.nInternalFaces();
    auto nCells = mesh.nCells();

    // start with one to include the diagonal
    auto nFacesPerCellH = Vector<localIdx>(SerialExecutor {}, nCells, 1);
    auto [neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH] =
        copyToHosts(neiOffs, ownOffs, diagOffs, mesh.faceOwner(), mesh.faceNeighbour());

    auto [nFacesPerCellHV, neiOffsetHV, ownOffsetHV, diagOffsetHV, faceOwnHV, faceNeiHV] =
        views(nFacesPerCellH, neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH);

    // accumulate number non-zeros per row
    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto own = faceOwnHV[facei];
            auto nei = faceNeiHV[facei];

            Kokkos::atomic_inc(&nFacesPerCellHV[own]);
            Kokkos::atomic_inc(&nFacesPerCellHV[nei]);
        },
        "setSparsityPatternFaceToMatrixAddress::accumulateNonZeros"
    );

    // get number of total non-zeros
    auto rowOffsH = rowOffs.copyToHost();
    auto rowOffsHV = rowOffsH.view();
    segmentsFromIntervals(nFacesPerCellH, rowOffsH);
    auto colIdxH = colIdx.copyToHost();
    auto colIdxHV = colIdxH.view();
    fill(nFacesPerCellH, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];

            // TODO this is probably inherently serial
            // return the oldValues
            // hit on performance on serial
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellHV[nei], 1);
            neiOffsetHV[facei] = static_cast<uint8_t>(segIdxNei);

            auto startSegNei = rowOffsHV[nei];
            // neighbour --> current cell
            // colIdx for row[neighbour] stores owner as a column entry
            Kokkos::atomic_store(&colIdxHV[startSegNei + segIdxNei], own);
        },
        "setSparsityPatternFaceToMatrixAddress::computeLowerTriangular"
    );

    map(
        nFacesPerCellH,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            // store number of lower entries as diagonal offset
            diagOffsetHV[celli] = static_cast<uint8_t>(nFaces);
            colIdxHV[rowOffsHV[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiHV[facei];
            auto own = faceOwnHV[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellHV[own], 1));
            ownOffsetHV[facei] = segIdxOwn;

            auto startSegOwn = rowOffsHV[own];
            // owner --> current cell
            // colIdx --> needs to be store the neighbour
            Kokkos::atomic_store(&colIdxHV[startSegOwn + segIdxOwn], nei);
        },
        "setSparsityPatternFaceToMatrixAddress::computeUpperTriangular"
    );

    // NOTE copy back to device
    const auto exec = mesh.exec();
    ownOffs = ownOffsetH.copyToExecutor(exec);
    neiOffs = neiOffsetH.copyToExecutor(exec);
    diagOffs = diagOffsetH.copyToExecutor(exec);
    colIdx = colIdxH.copyToExecutor(exec);
    rowOffs = rowOffsH.copyToExecutor(exec);
}

template<typename SparsityType>
std::pair<
    std::shared_ptr<const SparsityType>,
    std::shared_ptr<const FaceToMatrixAddress<typename SparsityType::SparsityIndexType>>>
createSparsityPatternFaceToMatrixAddress(const UnstructuredMesh& mesh)
{
    using IndexType = typename SparsityType::SparsityIndexType;
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<IndexType> rowOffs(exec, nCells + 1, 0);
    Vector<IndexType> colIdx(exec, nCells + 2 * nInternalFaces, 0);

    setSparsityPatternFaceToMatrixAddressSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);
    auto sp = std::make_shared<const SparsityType>(
        std::move(colIdx), std::move(rowOffs), Dimensions {nCells, nCells}
    );
    auto ftma = std::make_shared<const FaceToMatrixAddress<IndexType>>(ownOffs, neiOffs, diagOffs);
    return {sp, ftma};
}

// COO specialization: internal algorithm produces CSR-style rowOffs, so we must expand
// them to flat per-NNZ row indices as required by CooSparsityPattern.
template<>
std::pair<
    std::shared_ptr<const CooSparsityPattern<localIdx>>,
    std::shared_ptr<const FaceToMatrixAddress<localIdx>>>
createSparsityPatternFaceToMatrixAddress<CooSparsityPattern<localIdx>>(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    const localIdx nnz = nCells + 2 * nInternalFaces;
    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<localIdx> rowOffs(exec, nCells + 1, 0);
    Vector<localIdx> colIdx(exec, nnz, 0);

    setSparsityPatternFaceToMatrixAddressSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);

    // Convert CSR rowOffs → COO flat row indices
    Vector<localIdx> cooRowIdx(exec, nnz, 0);
    {
        auto rowOffsH = rowOffs.copyToHost();
        auto cooRowIdxH = cooRowIdx.copyToHost();
        const auto rowOffsHV = rowOffsH.view();
        auto cooRowIdxHV = cooRowIdxH.view();
        for (localIdx r = 0; r < nCells; r++)
            for (localIdx j = rowOffsHV[r]; j < rowOffsHV[r + 1]; j++)
                cooRowIdxHV[j] = r;
        cooRowIdx = cooRowIdxH.copyToExecutor(exec);
    }

    auto sp = std::make_shared<const CooSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(cooRowIdx), Dimensions {nCells, nCells}
    );
    auto ftma = std::make_shared<const FaceToMatrixAddress<localIdx>>(ownOffs, neiOffs, diagOffs);
    return {sp, ftma};
}

template<>
std::shared_ptr<const CooSparsityPattern<localIdx>>
createBoundarySparsityPattern<CooSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress<localIdx>& faceToMatrixAddress
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    Vector<localIdx> bRowOffs(exec, nBoundaryFaces, 0);
    Vector<localIdx> bColIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, faceToMatrixAddress.diagOffset(), bRowOffs, bColIdx);
    return std::make_shared<const CooSparsityPattern<localIdx>>(
        std::move(bColIdx), std::move(bRowOffs), Dimensions {mesh.nCells(), mesh.nCells()}
    );
}

template<>
std::shared_ptr<const CsrSparsityPattern<localIdx>>
createBoundarySparsityPattern<CsrSparsityPattern<localIdx>>(
    const UnstructuredMesh& mesh, const FaceToMatrixAddress<localIdx>& faceToMatrixAddress
)
{
    const auto exec = mesh.exec();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    Vector<localIdx> bRowOffs(exec, nBoundaryFaces, 0);
    Vector<localIdx> bColIdx(exec, nBoundaryFaces, 0);
    setBoundarySparsityPatternImpl(mesh, faceToMatrixAddress.diagOffset(), bRowOffs, bColIdx);
    return std::make_shared<const CsrSparsityPattern<localIdx>>(
        std::move(bColIdx), std::move(bRowOffs), Dimensions {mesh.nCells(), mesh.nCells()}
    );
}

template std::pair<
    std::shared_ptr<const CsrSparsityPattern<localIdx>>,
    std::shared_ptr<const FaceToMatrixAddress<localIdx>>>
createSparsityPatternFaceToMatrixAddress<CsrSparsityPattern<localIdx>>(const UnstructuredMesh&);

}
