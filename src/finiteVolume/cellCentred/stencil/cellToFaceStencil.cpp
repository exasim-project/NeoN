// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoN::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedVector<localIdx, localIdx> CellToFaceStencil::computeInternalStencil() const
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();
    const auto [faceOwners, faceNeighbors] = views(mesh_.faceOwners(), mesh_.faceNeighbors());

    const auto nInternalFaces = mesh_.nInternalFaces();

    const SerialExecutor serialExec;
    Vector<localIdx> nFacesPerCell(serialExec, nCells, 0);
    View<localIdx> nFacesPerCellView = nFacesPerCell.view();

    auto hostFaceOwners = mesh_.faceOwners().copyToHost();
    auto hostFaceNeighbors = mesh_.faceNeighbors().copyToHost();
    const auto [hostFaceOwnersView, hostFaceNeighborsView] =
        views(hostFaceOwners, hostFaceNeighbors);

    parallelFor(
        serialExec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Kokkos::atomic_inc(&nFacesPerCellView[hostFaceOwnersView[i]]);
            Kokkos::atomic_inc(&nFacesPerCellView[hostFaceNeighborsView[i]]);
        },
        "countFacesPerCellInternal"
    );

    SegmentedVector<localIdx, localIdx> stencil(nFacesPerCell);
    auto [stencilValues, segment] = stencil.views();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        serialExec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            localIdx owner = hostFaceOwnersView[facei];
            localIdx neigh = hostFaceNeighborsView[facei];

            const auto segIdxOwn = Kokkos::atomic_fetch_inc(&nFacesPerCellView[owner]);
            const auto segIdxNei = Kokkos::atomic_fetch_inc(&nFacesPerCellView[neigh]);

            auto segOwn = segment[owner] + segIdxOwn;
            auto segNei = segment[neigh] + segIdxNei;

            stencilValues[segOwn] = facei;
            stencilValues[segNei] = facei;
        },
        "computeStencilInternal"
    );

    return stencil.copyToExecutor(exec);
}

} // namespace NeoN::finiteVolume::cellCentred
