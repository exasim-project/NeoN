// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/DES/maxDeltaxyz.hpp"

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

// #include <Kokkos_Core.hpp>
// #include <algorithm>

namespace NeoN::turbulenceModels::DES
{

maxDeltaxyz::maxDeltaxyz(const UnstructuredMesh& mesh)
    : mesh_(mesh), delta_(mesh.exec(), mesh.nCells(), 0.0)
{
    update();
}

void maxDeltaxyz::update()
{
    const auto exec = mesh_.exec();
    const auto nCells = mesh_.nCells();

    Vector<scalar> minX(exec, nCells, 0.0);
    Vector<scalar> minY(exec, nCells, 0.0);
    Vector<scalar> minZ(exec, nCells, 0.0);
    Vector<scalar> maxX(exec, nCells, 0.0);
    Vector<scalar> maxY(exec, nCells, 0.0);
    Vector<scalar> maxZ(exec, nCells, 0.0);

    const auto faceCentres = mesh_.faceCentres().view();
    const auto cellCentres = mesh_.cellCentres().view();
    const auto owner = mesh_.faceOwner().view();
    const auto neighbour = mesh_.faceNeighbour().view();

    const auto minXView = minX.view();
    const auto minYView = minY.view();
    const auto minZView = minZ.view();
    const auto maxXView = maxX.view();
    const auto maxYView = maxY.view();
    const auto maxZView = maxZ.view();

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nFaces = mesh_.nFaces();

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const auto centre = cellCentres[celli];
            minXView[celli] = centre[0];
            minYView[celli] = centre[1];
            minZView[celli] = centre[2];
            maxXView[celli] = centre[0];
            maxYView[celli] = centre[1];
            maxZView[celli] = centre[2];
        },
        "maxDeltaxyz::initBounds"
    );

    auto updateCellBounds = NEON_LAMBDA(const localIdx celli, const Vec3& centre)
    {
        Kokkos::atomic_min(&minXView[celli], centre[0]);
        Kokkos::atomic_min(&minYView[celli], centre[1]);
        Kokkos::atomic_min(&minZView[celli], centre[2]);
        Kokkos::atomic_max(&maxXView[celli], centre[0]);
        Kokkos::atomic_max(&maxYView[celli], centre[1]);
        Kokkos::atomic_max(&maxZView[celli], centre[2]);
    };

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto faceCentre = faceCentres[facei];
            updateCellBounds(owner[facei], faceCentre);
            updateCellBounds(neighbour[facei], faceCentre);
        },
        "maxDeltaxyz::updateInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto faceCentre = faceCentres[facei];
            updateCellBounds(owner[facei], faceCentre);
        },
        "maxDeltaxyz::updateBoundary"
    );

    auto deltaView = delta_.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const scalar dx = maxXView[celli] - minXView[celli];
            const scalar dy = maxYView[celli] - minYView[celli];
            const scalar dz = maxZView[celli] - minZView[celli];
            deltaView[celli] = std::max(dx, std::max(dy, dz));
        },
        "maxDeltaxyz::updateDelta"
    );
}

const Vector<scalar>& maxDeltaxyz::delta() const { return delta_; }

} // namespace NeoN::turbulenceModels::DES
