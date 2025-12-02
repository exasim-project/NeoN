// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedVector<localIdx, localIdx> CellToFaceStencil::computeStencil() const
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto [faceOwner, faceNeighbour, boundaryFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto nInternalFaces = mesh_.nInternalFaces();

    Vector<localIdx> nFacesPerCell(exec, nCells, 0);
    View<localIdx> nFacesPerCellView = nFacesPerCell.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellView[static_cast<size_t>(faceOwner[i])]);
            Kokkos::atomic_increment(&nFacesPerCellView[static_cast<size_t>(faceNeighbour[i])]);
        },
        "countFacesPerCellInternal"
    );

    parallelFor(
        exec,
        {0, boundaryFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellView[boundaryFaceCells[i]]);
        },
        "countFacesPerCellBoundary"
    );

    SegmentedVector<localIdx, localIdx> stencil(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.views();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            localIdx owner = faceOwner[facei];
            localIdx neighbour = faceNeighbour[facei];

            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellView[owner], 1);
            localIdx segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellView[neighbour], 1);

            auto startSegOwn = segment[owner];
            auto startSegNei = segment[neighbour];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
            Kokkos::atomic_assign(&stencilValues[startSegNei + segIdxNei], facei);
        },
        "computeStencilInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + boundaryFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            localIdx owner = boundaryFaceCells[facei - nInternalFaces];
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellView[owner], 1);
            localIdx startSegOwn = segment[owner];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
        },
        "computeStencilBound"
    );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred
