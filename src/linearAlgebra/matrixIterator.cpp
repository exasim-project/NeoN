// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/matrixIterator.hpp"

namespace NeoN::la
{

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::ownerOffset() const
{
    return ownerOffset_;
}

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::neighbourOffset() const
{
    return neighbourOffset_;
}

template<typename IndexType, typename MeshType>
const NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::diagOffset() const
{
    return diagOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::ownerOffset()
{
    return ownerOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::neighbourOffset()
{
    return neighbourOffset_;
}

template<typename IndexType, typename MeshType>
NeoN::Array<uint8_t>& MatrixIterator<IndexType, MeshType>::diagOffset()
{
    return diagOffset_;
}

template<typename IndexType, typename MeshType>
MatrixIterator<IndexType, MeshType>::MatrixIterator(
    Array<uint8_t> ownerOffset,
    Array<uint8_t> neighbourOffset,
    Array<uint8_t> diagOffset,
    std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern,
    std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsityPattern
)
    : ownerOffset_(ownerOffset), neighbourOffset_(neighbourOffset), diagOffset_(diagOffset),
      ownerOffsetV_(ownerOffset_.view()), neighbourOffsetV_(neighbourOffset_.view()),
      diagOffsetV_(diagOffset_.view()), sp_(sparsityPattern), bsp_(boundarySparsityPattern)
{}

template<typename IndexType, typename MeshType>
MatrixIterator<IndexType, MeshType>::MatrixIterator(const MatrixIterator& mi)
    : ownerOffset_(mi.ownerOffset_), neighbourOffset_(mi.neighbourOffset_),
      diagOffset_(mi.diagOffset_), ownerOffsetV_(ownerOffset_.view()),
      neighbourOffsetV_(neighbourOffset_.view()), diagOffsetV_(diagOffset_.view()), sp_(mi.sp_),
      bsp_(mi.bsp_)
{}


template class MatrixIterator<localIdx, UnstructuredMesh>;

template<typename IndexType>
void setBoundarySparsityPattern(
    const UnstructuredMesh& mesh,
    Array<uint8_t>& diagOffs,
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
        "setSparsityPatternMatrixIterator::setBoundarySparsity"
    );
}

template<typename IndexType>
void setSparsityPatternMatrixIteratorSerial(
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
        "setSparsityPatternMatrixIterator::accumulateNonZeros"
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
        "setSparsityPatternMatrixIterator::computeLowerTriangular"
    );

    map(
        nFacesPerCellH,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            // store number of lower entries
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
        "setSparsityPatternMatrixIterator::computeUpperTriangular"
    );


    // NOTE copy back to device
    const auto exec = mesh.exec();
    ownOffs = ownOffsetH.copyToExecutor(exec);
    neiOffs = neiOffsetH.copyToExecutor(exec);
    diagOffs = diagOffsetH.copyToExecutor(exec);
    colIdx = colIdxH.copyToExecutor(exec);
    rowOffs = rowOffsH.copyToExecutor(exec);
}

template<typename IndexType>
MatrixIterator<IndexType> createSparsityPatternMatrixIterator(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.boundaryMesh().faceCells().size();
    const auto nCells = mesh.nCells();
    Array<uint8_t> diagOffs(exec, nCells, 0);
    Array<uint8_t> ownOffs(exec, nInternalFaces, 0);
    Array<uint8_t> neiOffs(exec, nInternalFaces, 0);
    Vector<IndexType> rowOffs(exec, nCells + 1, 0);
    Vector<IndexType> colIdx(exec, nCells + 2 * nInternalFaces, 0);
    Vector<IndexType> bRowOffs(exec, nBoundaryFaces, 0);
    Vector<IndexType> bColIdx(exec, nBoundaryFaces, 0);

    setSparsityPatternMatrixIteratorSerial(mesh, diagOffs, ownOffs, neiOffs, rowOffs, colIdx);
    auto sp =
        std::make_shared<const SparsityPattern<IndexType>>(std::move(colIdx), std::move(rowOffs));
    setBoundarySparsityPattern(mesh, diagOffs, bRowOffs, bColIdx);
    auto bsp =
        std::make_shared<const SparsityPattern<IndexType>>(std::move(bColIdx), std::move(bRowOffs));
    return MatrixIterator<IndexType>(ownOffs, neiOffs, diagOffs, sp, bsp);
}


template MatrixIterator<localIdx>
createSparsityPatternMatrixIterator<localIdx>(const UnstructuredMesh&);

}
