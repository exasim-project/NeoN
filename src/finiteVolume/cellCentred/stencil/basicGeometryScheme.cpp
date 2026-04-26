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
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    // NF_ASSERT(dstS.size() == ownerS.size(), "Inconsistent size");
    // NF_ASSERT(dstS.size() == neighS.size(), "Inconsistent size");
    // NF_ASSERT(dstS.size() == weightS.size(), "Inconsistent size");

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
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            weightS[facei] = 1.0;
            weightB[bcfacei] = 1.0;
        },
        "basicGeometricScheme::updateWeightsBoundary"
    );

    // Processor boundary: linear interpolation weight between the local owner cell
    // and the ghost cell on the neighbouring rank.
    //
    // For an internal face the weight is
    //   w = sfdNei / (sfdOwn + sfdNei)    with    sfdOwn = |Sf · (cf - c_own)|,
    //                                              sfdNei = |Sf · (c_nei - cf)|,
    // matching `phi_f = w * phi_own + (1 - w) * phi_nei`.
    //
    // For processor faces we don't have c_nei locally (the ghost cell center is on
    // another rank). Without an extra MPI exchange of cell centers, the consistent
    // convention with updateNonOrthDeltaCoeffs / updateDeltaCoeffs (which both use
    // the face-as-midpoint assumption — see the 0.5/orthoDist factor below) is
    // sfdNei == sfdOwn, giving w = 0.5. That matches what processorPolyPatch::
    // makeWeights produces on a uniformly decomposed mesh and gives the symmetric
    // linear interpolation `phi_f = 0.5 * phi_own + 0.5 * phi_ghost` consumed by
    // `flux()` and the implicit divergence operator's proc-boundary handling.
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            weightS[facei] = 0.5;
            weightB[bcfacei] = 0.5;
        },
        "basicGeometricScheme::updateWeightsProcBoundary"
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
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCentre[neighbour[facei]] - cellCentre[owner[facei]];
            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsInternal"
    );

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];

            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
        },
        "basicGeometricScheme::updateDeltaCoeffsBoundary"
    );

    // FIXME
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];

            deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
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
    fill(nonOrthDeltaCoeffs.internalVector(), 0.0);

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto totalFaces = mesh_.nTotalFaces();

    NeoN::mpi::Environment mpiEnviron;
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
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );

    // FIXME
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = surfFaceCells[facei - nInternalFaces];
            Vec3 cellToCellDist = cf[facei] - cellCentre[own];
            Vec3 faceNormal = 1 / faceArea[facei] * faceAreaVec3[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 0.5 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vec3>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoN
