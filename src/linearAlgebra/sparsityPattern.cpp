// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

const SparsityPattern& SparsityPattern::readOrCreate(const UnstructuredMesh& mesh)
{
    StencilDataBase& stencilDb = mesh.stencilDB();
    if (!stencilDb.contains("SparsityPattern"))
    {
        stencilDb.insert(std::string("SparsityPattern"), SparsityPattern(mesh));
    }
    return stencilDb.get<SparsityPattern>("SparsityPattern");
}

void updateSparsityPattern(const UnstructuredMesh& mesh, SparsityPattern& sp)
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
        KOKKOS_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto owner = faceOwner[facei];
            auto neighbour = faceNeiV[facei];

            Kokkos::atomic_increment(&nFacesPerCellView[owner]);
            Kokkos::atomic_increment(&nFacesPerCellView[neighbour]);
        }
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
        KOKKOS_LAMBDA(const localIdx facei) {
            auto neighbour = faceNeiV[facei];
            auto owner = faceOwner[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellView[neighbour], 1);
            neighbourOffsetView[facei] = static_cast<uint8_t>(segIdxNei);

            auto startSegNei = rowOffs[neighbour];
            // neighbour --> current cell
            // colIdx --> needs to be store the owner
            Kokkos::atomic_assign(&colIdxV[startSegNei + segIdxNei], owner);
        }
    );

    map(
        nFacesPerCell,
        KOKKOS_LAMBDA(const localIdx celli) {
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
        KOKKOS_LAMBDA(const localIdx facei) {
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
            Kokkos::atomic_assign(&colIdxV[startSegOwn + segIdxOwn], neighbour);
        }
    );
}

SparsityPattern createSparsity(const UnstructuredMesh& mesh)
{
    const auto exec = mesh.exec();
    const auto nCells = mesh.nCells();
    const auto nnzs = 2 * mesh.nInternalFaces() + nCells;
    auto ret = SparsityPattern(exec, nCells, nnzs);
    updateSparsityPattern(mesh, ret);
    return ret;
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


SparsityPattern::SparsityPattern(Executor exec, localIdx nRows, localIdx nnzs)
    : rowOffs_(exec, nRows + 1, 0), colIdxs_(exec, nnzs, 0),
      ownerOffset_(exec, (nnzs - nRows) / 2, 0), neighbourOffset_(exec, (nnzs - nRows) / 2, 0),
      diagOffset_(exec, nRows, 0)
{}

const NeoN::Array<uint8_t>& SparsityPattern::ownerOffset() const { return ownerOffset_; }

const NeoN::Array<uint8_t>& SparsityPattern::neighbourOffset() const { return neighbourOffset_; }

const NeoN::Array<uint8_t>& SparsityPattern::diagOffset() const { return diagOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::ownerOffset() { return ownerOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::neighbourOffset() { return neighbourOffset_; }

NeoN::Array<uint8_t>& SparsityPattern::diagOffset() { return diagOffset_; }

} // namespace NeoN::la
