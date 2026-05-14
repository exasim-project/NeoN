// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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
    const auto [faceOwnV, faceNeiV, faceFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto nInternalFaces = mesh_.nInternalFaces();

    auto nFacesPerCell = Vector<localIdx>(exec, nCells, 0);
    auto nFacesPerCellV = nFacesPerCell.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Kokkos::atomic_inc(&nFacesPerCellV[faceOwnV[i]]);
            Kokkos::atomic_inc(&nFacesPerCellV[faceNeiV[i]]);
        },
        "countFacesPerCellInternal"
    );

    parallelFor(
        exec,
        {0, faceFaceCells.size()},
        NEON_LAMBDA(const localIdx i) { Kokkos::atomic_inc(&nFacesPerCellV[faceFaceCells[i]]); },
        "countFacesPerCellBoundary"
    );

    auto stencil = SegmentedVector<localIdx, localIdx>(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.views();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto nei = faceNeiV[facei]; // neighbour cell idx
            auto own = faceOwnV[facei]; // owning cell idx

            // obtain the old values and increment
            localIdx segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellV[nei], 1);
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellV[own], 1);
            auto startSegNei = segment[nei];
            auto startSegOwn = segment[own];
            Kokkos::atomic_store(&stencilValues[startSegNei + segIdxNei], facei);
            Kokkos::atomic_store(&stencilValues[startSegOwn + segIdxOwn], facei);
        },
        "computeStencilInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + faceFaceCells.size()},
        NEON_LAMBDA(const localIdx facei) {
            auto owner = faceFaceCells[facei - nInternalFaces];
            // obtain the old values and increment
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellV[owner], 1);
            auto startSegOwn = segment[owner];
            Kokkos::atomic_store(&stencilValues[startSegOwn + segIdxOwn], facei);
        },
        "computeStencilBound"
    );

    // sort face ids in stencil to be in face order
    // parallelFor(
    //     exec,
    //     {0, segment.size()},
    //     NEON_LAMBDA(const localIdx celli) { auto nCells = nFacesPerCellV[celli]; }
    // );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred
