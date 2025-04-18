// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/fields/segmentedVector.hpp"

namespace NeoN::finiteVolume::cellCentred
{

SparsityPattern::SparsityPattern(const UnstructuredMesh& mesh)
    : mesh_(mesh), rowPtrs_(mesh_.exec(), mesh.nCells() + 1, 0),
      colIdxs_(mesh_.exec(), mesh.nCells() + 2 * mesh.nInternalFaces(), 0),
      ownerOffset_(mesh_.exec(), mesh_.nInternalFaces(), 0),
      neighbourOffset_(mesh_.exec(), mesh_.nInternalFaces(), 0),
      diagOffset_(mesh_.exec(), mesh_.nCells(), 0)
{
    update();
}

const std::shared_ptr<SparsityPattern> SparsityPattern::readOrCreate(const UnstructuredMesh& mesh)
{
    StencilDataBase& stencilDb = mesh.stencilDB();
    if (!stencilDb.contains("SparsityPattern"))
    {
        stencilDb.insert(std::string("SparsityPattern"), std::make_shared<SparsityPattern>(mesh));
    }
    return stencilDb.get<std::shared_ptr<SparsityPattern>>("SparsityPattern");
}

void SparsityPattern::update()
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto faceOwner = mesh_.faceOwner().view();
    const auto faceNeighbour = mesh_.faceNeighbour().view();
    // const auto faceFaceCells = mesh_.boundaryMesh().faceCells().view();
    const auto nInternalFaces = mesh_.nInternalFaces();

    // start with one to include the diagonal
    Vector<localIdx> nFacesPerCell(exec, nCells, 1);
    auto [nFacesPerCellSpan, neighbourOffsetSpan, ownerOffsetSpan, diagOffsetSpan] =
        spans(nFacesPerCell, neighbourOffset_, ownerOffset_, diagOffset_);

    // accumulate number non-zeros per row
    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            // hit on performance on serial
            auto owner = faceOwner[facei];
            auto neighbour = faceNeighbour[facei];

            Kokkos::atomic_increment(&nFacesPerCellSpan[owner]);
            Kokkos::atomic_increment(&nFacesPerCellSpan[neighbour]);
        }
    );

    // get number of total non-zeros
    segmentsFromIntervals(nFacesPerCell, rowPtrs_);
    auto rowPtrs = rowPtrs_.view();
    View<localIdx> sColIdx = colIdxs_.view();
    fill(nFacesPerCell, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto neighbour = faceNeighbour[facei];
            auto owner = faceOwner[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[neighbour], 1);
            neighbourOffsetSpan[facei] = static_cast<uint8_t>(segIdxNei);

            auto startSegNei = rowPtrs[neighbour];
            // neighbour --> current cell
            // colIdx --> needs to be store the owner
            Kokkos::atomic_assign(&sColIdx[startSegNei + segIdxNei], owner);
        }
    );

    map(
        nFacesPerCell,
        KOKKOS_LAMBDA(const localIdx celli) {
            auto nFaces = nFacesPerCellSpan[celli];
            diagOffsetSpan[celli] = static_cast<uint8_t>(nFaces);
            sColIdx[rowPtrs[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto neighbour = faceNeighbour[facei];
            auto owner = faceOwner[facei];

            // return the oldValues
            // hit on performance on serial
            auto segIdxOwn =
                static_cast<uint8_t>(Kokkos::atomic_fetch_add(&nFacesPerCellSpan[owner], 1));
            ownerOffsetSpan[facei] = segIdxOwn;

            auto startSegOwn = rowPtrs[owner];
            // owner --> current cell
            // colIdx --> needs to be store the neighbour
            Kokkos::atomic_assign(&sColIdx[startSegOwn + segIdxOwn], neighbour);
        }
    );
}


const NeoN::Vector<uint8_t>& SparsityPattern::ownerOffset() const { return ownerOffset_; }

const NeoN::Vector<uint8_t>& SparsityPattern::neighbourOffset() const { return neighbourOffset_; }

const NeoN::Vector<uint8_t>& SparsityPattern::diagOffset() const { return diagOffset_; }

} // namespace NeoN::finiteVolume::cellCentred
