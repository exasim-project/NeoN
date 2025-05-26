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
    const auto [faceOwnV, faceNeiV, faceFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto nInternalFaces = mesh_.nInternalFaces();

    auto nFacesPerCell = Vector<localIdx>(exec, nCells, 0);
    auto nFacesPerCellV = nFacesPerCell.view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellV[faceOwnV[i]]); // hit on performance on serial
            Kokkos::atomic_increment(&nFacesPerCellV[faceNeiV[i]]);
        }
    );

    parallelFor(
        exec,
        {0, faceFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            Kokkos::atomic_increment(&nFacesPerCellV[faceFaceCells[i]]);
        }
    );

    auto stencil = SegmentedVector<localIdx, localIdx>(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.views();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    Kokkos::fence();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto nei = faceNeiV[facei]; // neighbour cell idx
            auto own = faceOwnV[facei]; // owning cell idx
            // obtain the old values and increment
            localIdx segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellV[nei], 1);
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellV[own], 1);
            auto startSegNei = segment[nei];
            auto startSegOwn = segment[own];
            Kokkos::atomic_assign(&stencilValues[startSegNei + segIdxNei], facei);
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
        }
    );

    Kokkos::fence();
    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + faceFaceCells.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto owner = faceFaceCells[facei - nInternalFaces];
            // obtain the old values and increment
            localIdx segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellV[owner], 1);
            auto startSegOwn = segment[owner];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
        }
    );

    // sort face ids in stencil to be in face order
    parallelFor(
        exec,
        {0, segment.size()},
        KOKKOS_LAMBDA(const localIdx celli) { auto nCells = nFacesPerCellV[celli]; }
    );

    return stencil;
}

} // namespace NeoN::finiteVolume::cellCentred
