// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

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
        NEON_LAMBDA(const localIdx facei) {
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
        },
        "basicGeometricScheme::updateWeightsInternal"
    );

    parallelFor(
        exec,
        {0, mesh_.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bfi) { weightB[bfi] = 1.0; },
        "basicGeometricScheme::updateWeightsBoundary"
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
    auto deltaCoeffB = deltaCoeffs.boundaryData().value().view();

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsInternal"
    );

    parallelFor(
        exec,
        {0, mesh_.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = surfFaceCells[bfi];
            Vec3 cellToCellDist = cf[nInternalFaces + bfi] - cellCentre[own];
            deltaCoeffB[bfi] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsBoundary"
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
    auto nonOrthDeltaCoeffB = nonOrthDeltaCoeffs.boundaryData().value().view();
    fill(nonOrthDeltaCoeffs.internalVector(), 0.0);

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsInternal"
    );

    parallelFor(
        exec,
        {0, mesh_.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = surfFaceCells[bfi];
            Vec3 cellToCellDist = cf[nInternalFaces + bfi] - cellCentre[own];
            Vec3 faceNormal =
                1 / faceArea[nInternalFaces + bfi] * faceAreaVec3[nInternalFaces + bfi];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeffB[bfi] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );
}


void BasicGeometryScheme::updateNonOrthCorrectionVec3s(
    const Executor& exec, SurfaceField<Vec3>& nonOrthCorrectionVec3s
)
{
    const auto [owner, neighbour, surfFaceCells] =
        views(mesh_.faceOwner(), mesh_.faceNeighbour(), mesh_.boundaryMesh().faceCells());

    const auto [cellCentre, faceAreaVec3, faceArea] =
        views(mesh_.cellCentres(), mesh_.faceAreas(), mesh_.magFaceAreas());

    const auto [corrVec, corrVecB] = views(
        nonOrthCorrectionVec3s.internalVector(), nonOrthCorrectionVec3s.boundaryData().value()
    );

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 delta = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            Vec3 n = (1.0 / faceArea[facei]) * faceAreaVec3[facei];
            scalar orthoDist = n & delta;
            scalar nonOrthDeltaCoeff = 1.0 / std::max(orthoDist, scalar(0.05) * mag(delta));
            corrVec[facei] = n - delta * nonOrthDeltaCoeff;
        },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sInternal"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { corrVecB[bfi] = zero<Vec3>(); },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sBoundary"
    );
}

} // namespace NeoN
