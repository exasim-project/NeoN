// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

template<typename IndexType>
void SparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowOffs_.exec() == colIdxs_.exec(), "Executors are not the same");
}

template<typename IndexType>
SparsityPattern<IndexType>::SparsityPattern(Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs)
    : rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx)), rowOffsV_(rowOffs_.view()),
      colIdxsV_(colIdxs_.view())
{
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = mesh.exec();
    auto nCells = mesh.nCells();

    // start with one to include the diagonal
    // TODO: currently the whole algorithm is performed in serial on the host
    auto nFacesPerCellH = Vector<localIdx>(SerialExecutor {}, nCells, 1);
    auto [neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH] = copyToHosts(
        sp.neighbourOffset(),
        sp.ownerOffset(),
        sp.diagOffset(),
        mesh.faceOwner(),
        mesh.faceNeighbour()
    );

    auto [nFacesPerCellHV, neiOffsetHV, ownOffsetHV, diagOffsetHV, faceOwnHV, faceNeiHV] =
        views(nFacesPerCellH, neiOffsetH, ownOffsetH, diagOffsetH, faceOwnH, faceNeiH);

    // accumulate number non-zeros per row
    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto own = faceOwnHV[facei];
            auto nei = faceNeiHV[facei];

            Kokkos::atomic_inc(&nFacesPerCellHV[own]);
            Kokkos::atomic_inc(&nFacesPerCellHV[nei]);
        },
        "updateSparsityPattern::accumulateNonZeros"
    );

    // get number of total non-zeros
    auto rowOffsH = sp.rowOffs().copyToHost();
    auto rowOffsHV = rowOffsH.view();
    segmentsFromIntervals(nFacesPerCellH, rowOffsH);
    auto colIdxH = sp.colIdxs().copyToHost();
    auto colIdxHV = colIdxH.view();
    fill(nFacesPerCellH, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
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
        "updateSparsityPattern::computeLowerTriangular"
    );

    map(
        nFacesPerCellH,
        NEON_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellHV[celli];
            diagOffsetHV[celli] = static_cast<uint8_t>(nFaces);
            colIdxHV[rowOffsHV[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        SerialExecutor {},
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
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
        "updateSparsityPattern::computeUpperTriangular"
    );
    // NOTE copy back to device
    sp.ownerOffset() = ownOffsetH.copyToExecutor(exec);
    sp.neighbourOffset() = neiOffsetH.copyToExecutor(exec);
    sp.diagOffset() = diagOffsetH.copyToExecutor(exec);
    sp.colIdxs() = colIdxH.copyToExecutor(exec);
    sp.rowOffs() = rowOffsH.copyToExecutor(exec);
}

void updateSparsityPatternParallel(const UnstructuredMesh& mesh, SparsityPattern& sp)
{
    const auto faceOwner = mesh.faceOwner().view();
    const auto faceNeiV = mesh.faceNeighbour().view();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto exec = mesh.exec();
    auto nCells = mesh.nCells();

    // start with one to include the diagonal
    auto nFacesPerCell = Vector<localIdx>(exec, nCells, 1);
    auto [nFacesPerCellView, neighbourOffsetView, ownerOffsetView, diagOffsetView] =
        views(nFacesPerCell, sp.neighbourOffset(), sp.ownerOffset(), sp.diagOffset());

    // accumulate number non-zeros per row
    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto owner = faceOwner[facei];
            auto neighbour = faceNeiV[facei];

            Kokkos::atomic_inc(&nFacesPerCellView[owner]);
            Kokkos::atomic_inc(&nFacesPerCellView[neighbour]);
        },
        "updateSparsityPatternParallel::accumulateNonZerojs"
    );

    // get number of total non-zeros
    auto rowOffs = sp.rowOffs().view();
    segmentsFromIntervals(nFacesPerCell, sp.rowOffs());
    auto colIdxV = sp.colIdxs().view();
    fill(nFacesPerCell, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto neighbour = faceNeiV[facei];
            auto owner = faceOwner[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellView[neighbour], 1);
            neighbourOffsetView[facei] = static_cast<uint8_t>(segIdxNei);

            auto startSegNei = rowOffs[neighbour];
            // neighbour --> current cell
            // colIdx --> needs to be store the owner
            Kokkos::atomic_store(&colIdxV[startSegNei + segIdxNei], owner);
        },
        "updateSparsityPatternParallel::computeLowerTriangular"
    );

    map(
        nFacesPerCell,
        NEON_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellView[celli];
            diagOffsetView[celli] = static_cast<uint8_t>(nFaces);
            colIdxV[rowOffs[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto neighbour = faceNeiV[facei];
            auto owner = faceOwner[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellView[owner], 1));
            ownerOffsetView[facei] = segIdxOwn;

            auto startSegOwn = rowOffs[owner];
            // owner --> current cell
            // colIdx --> needs to be store the neighbour
            Kokkos::atomic_store(&colIdxV[startSegOwn + segIdxOwn], neighbour);
        },
        "updateSparsityPatternParallel::computeUpperTriangular"
    );
}

void updateSparsityPattern(const UnstructuredMesh& mesh, SparsityPattern& sp)
{
    // TODO sort pattern after creation and use parallel version
    updateSparsityPatternSerial(mesh, sp);
}


SparsityPattern::SparsityPattern(const UnstructuredMesh& mesh)
    : rowOffs_(mesh.exec(), mesh.nCells() + 1, 0),
      colIdxs_(mesh.exec(), mesh.nCells() + 2 * mesh.nInternalFaces(), 0),
      ownerOffset_(mesh.exec(), mesh.nInternalFaces(), 0),
      neighbourOffset_(mesh.exec(), mesh.nInternalFaces(), 0),
      diagOffset_(mesh.exec(), mesh.nCells(), 0)
{
    updateSparsityPattern(mesh, *this);
}

SparsityPattern::SparsityPattern(
    Vector<localIdx>&& rowOffs,
    Vector<localIdx>&& colIdx,
    Array<uint8_t>&& ownerOffset,
    Array<uint8_t>&& neighbourOffset,
    Array<uint8_t>&& diagOffset
)
    : rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx)),
      ownerOffset_(std::move(ownerOffset)), neighbourOffset_(std::move(neighbourOffset)),
      diagOffset_(std::move(neighbourOffset)) {};

/*@brief given colIdx and rowOffs this functions computes the corresponding offsets within a matrix
 * row*/
std::tuple<Array<uint8_t>, Array<uint8_t>, Array<uint8_t>>
computeOffsets(const Vector<localIdx>& colIdxs, const Vector<localIdx>& rowOffs)
{
    auto exec = rowOffs.exec();
    auto nnzs = colIdxs.size();

    auto ownerOffset = Array<uint8_t>(exec, nnzs); //! mapping from faceId to lower index in a row
    auto neighbourOffset =
        Array<uint8_t>(exec, nnzs);               //! mapping from faceId to upper index in a row
    auto diagOffset = Array<uint8_t>(exec, nnzs); //! mapping from faceId to column index in a row

    // FIXME
    // parallelFor(
    //     SerialExecutor {},
    //     {0, nInternalFaces},
    //     KOKKOS_LAMBDA(const localIdx facei) {


    //     },
    //     "computeOffsets::computeOwnerOffset"
    // );

    // parallelFor(
    //     SerialExecutor {},
    //     {0, nInternalFaces},
    //     KOKKOS_LAMBDA(const localIdx facei) {

    //     },
    //     "computeOffsets::computediagOffset"
    // );

    // parallelFor(
    //     SerialExecutor {},
    //     {0, nInternalFaces},
    //     KOKKOS_LAMBDA(const localIdx facei) {
    //     },
    //     "computeOffsets::computediagOffset"
    // );


    return {ownerOffset, neighbourOffset, diagOffset};
}

SparsityPattern::SparsityPattern(
    Executor exec, const Vector<localIdx>& colIdxs, const Vector<localIdx>& rowOffs
)
    : rowOffs_(rowOffs), colIdxs_(colIdxs), ownerOffset_(exec, int(1), 0),
      neighbourOffset_(exec, int(1), 0), diagOffset_(exec, rowOffs.size(), 0)
{
    auto [ownOffs, neighOffs, diagOffs] = computeOffsets(colIdxs, rowOffs);
    ownerOffset_ = ownOffs;
    neighbourOffset_ = neighOffs;
    diagOffset_ = diagOffs;
}

SparsityPattern::SparsityPattern(const SparsityPattern& sp)
    : exec_(sp.exec_), rowOffs_(sp.rowOffs_), colIdxs_(sp.colIdxs_), ownerOffset_(sp.ownerOffset_),
      neighbourOffset_(sp.neighbourOffset_), diagOffset_(sp.diagOffset_)
{}


#define NN_DECLARE_SPARSITY(TYPENAME) template class SparsityPattern<TYPENAME>

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
