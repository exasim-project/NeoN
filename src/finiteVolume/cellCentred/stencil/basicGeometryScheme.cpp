// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
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

    const auto [weightS, weightB] = views(weights.internalVector(), weights.boundaryData().value());
    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            scalar sfdOwn = std::abs(sf[facei] & (cf[facei] - c[owner[facei]]));
            scalar sfdNei = std::abs(sf[facei] & (c[neighbour[facei]] - cf[facei]));

            if (std::abs(sfdOwn + sfdNei) > ROOTVSMALL)
            {
                weightS[facei] = sfdNei / (sfdOwn + sfdNei);
            }
            else
            {
                weightS[facei] = 0.5;
            }
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, weightS.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            weightS[facei] = 1.0;
            weightB[bcfacei] = 1.0;
        }
    );
}

void BasicGeometryScheme::updateDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& deltaCoeffs
)
{
    const auto [owner, neighbour, surfFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());


    const auto [cf, cellCentre] = views(mesh_.faceCentres(), mesh_.cellCentres());

    auto deltaCoeff = deltaCoeffs.internalVector().view();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        KOKKOS_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        }
    );

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {nInternalFaces, deltaCoeff.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];

            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        }
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owner, neighbour, surfFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());


    const auto [cf, cellCentre, faceAreaVec3, faceArea] =
        views(mesh_.faceCentres(), mesh_.cellCentres(), mesh_.faceAreas(), mesh_.magFaceAreas());

    auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalVector().view();
    fill(nonOrthDeltaCoeffs.internalVector(), 0.0);

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];

            scalar orthoDist = faceNormal & cellToCellDist;


            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nonOrthDeltaCoeff.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];

            scalar orthoDist = faceNormal & cellToCellDist;


            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        }
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vec3>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoN
