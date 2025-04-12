// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"

namespace NeoN::finiteVolume::cellCentred
{

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(mesh), mesh_(mesh)
{}

void BasicGeometryScheme::updateWeights(const Executor& exec, SurfaceField<scalar>& weights)
{
    const auto owner = mesh_.faceOwner().view();
    const auto neighbour = mesh_.faceNeighbour().view();

    const auto cf = mesh_.faceCentres().view();
    const auto c = mesh_.cellCentres().view();
    const auto sf = mesh_.faceAreas().view();

    auto w = weights.internalField().view();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        KOKKOS_LAMBDA(const size_t facei) {
            scalar sfdOwn =
                std::abs(sf[facei] & (cf[facei] - c[static_cast<size_t>(owner[facei])]));
            scalar sfdNei =
                std::abs(sf[facei] & (c[static_cast<size_t>(neighbour[facei])] - cf[facei]));

            if (std::abs(sfdOwn + sfdNei) > ROOTVSMALL)
            {
                w[facei] = sfdNei / (sfdOwn + sfdNei);
            }
            else
            {
                w[facei] = 0.5;
            }
        }
    );

    parallelFor(
        exec,
        {mesh_.nInternalFaces(), w.size()},
        KOKKOS_LAMBDA(const size_t facei) { w[facei] = 1.0; }
    );
}

void BasicGeometryScheme::updateDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& deltaCoeffs
)
{
    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());


    const auto [cf, cellCentre] = spans(mesh_.faceCentres(), mesh_.cellCentres());

    auto deltaCoeff = deltaCoeffs.internalField().view();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        KOKKOS_LAMBDA(const size_t facei) {
            Vector cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        }
    );

    const size_t nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {nInternalFaces, deltaCoeff.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto own = static_cast<size_t>(surfFaceCells[facei - nInternalFaces]);
            Vector cellToCellDist = cf[facei] - cellCentre[own];

            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        }
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owner, neighbour, surfFaceCells] =
        spans(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());


    const auto [cf, cellCentre, faceAreaVector, faceArea] =
        spans(mesh_.faceCentres(), mesh_.cellCentres(), mesh_.faceAreas(), mesh_.magFaceAreas());

    auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalField().view();
    fill(nonOrthDeltaCoeffs.internalField(), 0.0);

    const size_t nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            Vector cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            Vector faceNormal = 1 / faceArea[facei] * faceAreaVector[facei];

            scalar orthoDist = faceNormal & cellToCellDist;


            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nonOrthDeltaCoeff.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            auto own = static_cast<size_t>(surfFaceCells[facei - nInternalFaces]);
            Vector cellToCellDist = cf[facei] - cellCentre[own];
            Vector faceNormal = 1 / faceArea[facei] * faceAreaVector[facei];

            scalar orthoDist = faceNormal & cellToCellDist;


            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        }
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vector>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoN
