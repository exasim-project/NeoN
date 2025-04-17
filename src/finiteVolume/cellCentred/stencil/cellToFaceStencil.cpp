// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedVector<localIdx, localIdx> CellToFaceStencil::computeStencil() const
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto [faceOwner, faceNeighbour, faceFaceCells] =
        spans(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto nInternalFaces = mesh_.nInternalFaces();

    Vector<localIdx> nFacesPerCell(exec, nCells, 0);
    View<localIdx> nFacesPerCellSpan = nFacesPerCell.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceOwner[i])]
            ); // hit on performance on serial
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceNeighbour[i])]);
        }
    );

    parallelFor(
        exec,
        {0, faceFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellSpan[faceFaceCells[i]]);
        }
    );

    SegmentedVector<localIdx, localIdx> stencil(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.spans();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            localIdx owner = faceOwner[facei];
            localIdx neighbour = faceNeighbour[facei];

            // return the oldValues
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(
                &nFacesPerCellSpan[owner], 1
            ); // hit on performance on serial
            localIdx segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[neighbour], 1);

            auto startSegOwn = segment[owner];
            auto startSegNei = segment[neighbour];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
            Kokkos::atomic_assign(&stencilValues[startSegNei + segIdxNei], facei);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + faceFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            localIdx owner = faceFaceCells[facei - nInternalFaces];
            // return the oldValues
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(
                &nFacesPerCellSpan[owner], 1
            ); // hit on performance on serial
            localIdx startSegOwn = segment[owner];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
        }
    );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred
