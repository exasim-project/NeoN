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

    // Proc boundary weights: w = d_N/(d_P+d_N) = 1 - d_P*deltaCoeffs
    // d_P = |patch.delta()| (owner cell to face centre, local), deltaCoeffs = 1/(d_P+d_N)
    // (d_N is on the remote rank; both d_P and d_P+d_N are available from BoundaryMesh).
    const auto bMeshDeltaV = mesh_.boundaryMesh().delta().view();
    const auto bMeshDeltaCoeffsV_w = mesh_.boundaryMesh().deltaCoeffs().view();
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            weightS[facei] = 1.0 - mag(bMeshDeltaV[bcfacei]) * bMeshDeltaCoeffsV_w[bcfacei];
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

    // Proc boundary delta coefficients: use the pre-computed values from the boundary mesh.
    // The neighbour cell centre is on a remote rank; use the OpenFOAM-provided deltaCoeffs
    // which account for both sides of the proc boundary.
    const auto bMeshDeltaCoeffsV = mesh_.boundaryMesh().deltaCoeffs().view();
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            deltaCoeff[facei] = bMeshDeltaCoeffsV[bcfacei];
        },
        "basicGeometricScheme::updateDeltaCoeffsProcBoundary"
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

    // Proc boundary non-orthogonal delta coefficients: use the pre-computed deltaCoeffs
    // from the boundary mesh (same as deltaCoeffs since non-orthogonal correction requires
    // the neighbour cell centre which is on a remote rank).
    const auto bMeshDeltaCoeffsV2 = mesh_.boundaryMesh().deltaCoeffs().view();
    parallelFor(
        exec,
        {nInternalFaces + nBoundaryFaces, totalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            nonOrthDeltaCoeff[facei] = bMeshDeltaCoeffsV2[bcfacei];
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsProcBoundary"
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vec3>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoN
