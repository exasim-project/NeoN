// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#endif

namespace NeoN::finiteVolume::cellCentred
{

#ifdef NF_WITH_MPI_SUPPORT
namespace
{

/** @brief Returns the [start, end) face-index ranges of all processor boundary patches in
 *  the order they appear in the boundary mesh offset array. */
std::vector<std::pair<localIdx, localIdx>> collectProcPatchRanges(const UnstructuredMesh& mesh)
{
    const auto& bMesh = mesh.boundaryMesh();
    const auto& off = bMesh.offset();
    const auto nBounds = bMesh.nBoundaries();
    const auto nProcPatches = bMesh.nProcBoundaryPatches();

    std::vector<std::pair<localIdx, localIdx>> ranges;
    ranges.reserve(static_cast<std::size_t>(nProcPatches));
    for (localIdx i = nBounds - nProcPatches; i < nBounds; ++i)
        ranges.push_back({off[static_cast<std::size_t>(i)], off[static_cast<std::size_t>(i + 1)]});
    return ranges;
}

/** @brief Computes the face-normal projection of the owner-cell-to-face distance for each
 *  processor boundary face, exchanges these distances with the neighbouring ranks via
 *  non-blocking MPI, and returns the received neighbour distances as a device Vector of
 *  size nProcBoundaryFaces. */
Vector<scalar> exchangeProcOwnerDistance(const Executor& exec, const UnstructuredMesh& mesh)
{
    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcFaces = mesh.nProcBoundaryFaces();

    auto bFaceCentersH = bMesh.faceCenters().copyToHost();
    auto bFaceNormalsH = bMesh.faceNormals().copyToHost();
    auto bFaceAreasH = bMesh.faceAreas().copyToHost();
    auto bFaceOwnersH = bMesh.faceOwners().copyToHost();
    auto cellCentersH = mesh.cellCenters().copyToHost();

    const auto bFaceCenters = bFaceCentersH.view();
    const auto bFaceNormals = bFaceNormalsH.view();
    const auto bFaceAreas = bFaceAreasH.view();
    const auto bFaceOwners = bFaceOwnersH.view();
    const auto cellCenters = cellCentersH.view();

    std::vector<scalar> dOwn(static_cast<std::size_t>(nProcFaces));
    std::vector<scalar> dNei(static_cast<std::size_t>(nProcFaces), scalar(0));
    for (localIdx i = 0; i < nProcFaces; ++i)
    {
        const localIdx bfi = nBoundaryFaces + i;
        const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
        dOwn[static_cast<std::size_t>(i)] =
            std::abs(n & (bFaceCenters[bfi] - cellCenters[bFaceOwners[bfi]]));
    }

    const auto ranges = collectProcPatchRanges(mesh);
    std::vector<MPI_Request> requests(2 * ranges.size(), MPI_REQUEST_NULL);
    mpi::Environment mpiEnv;
    for (std::size_t p = 0; p < ranges.size(); ++p)
    {
        const auto [rangeStart, rangeEnd] = ranges[p];
        const localIdx patchOff = rangeStart - nBoundaryFaces;
        const auto neighborRank = static_cast<mpi_label_t>(bMesh.neighbourRankForRange(ranges[p]));
        const auto byteCount = static_cast<mpi_label_t>((rangeEnd - rangeStart) * sizeof(scalar));
        mpi::isend<char>(
            reinterpret_cast<const char*>(dOwn.data() + patchOff),
            byteCount,
            neighborRank,
            0,
            mpiEnv.comm(),
            &requests[2 * p]
        );
        mpi::irecv<char>(
            reinterpret_cast<char*>(dNei.data() + patchOff),
            byteCount,
            neighborRank,
            0,
            mpiEnv.comm(),
            &requests[2 * p + 1]
        );
    }
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    Vector<scalar> result(SerialExecutor {}, nProcFaces, scalar(0));
    auto resultView = result.view();
    for (localIdx i = 0; i < nProcFaces; ++i)
        resultView[i] = dNei[static_cast<std::size_t>(i)];
    return result.copyToExecutor(exec);
}

} // anonymous namespace
#endif

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(mesh), mesh_(mesh)
{}

void BasicGeometryScheme::updateWeights(const Executor& exec, SurfaceField<scalar>& weights)
{
    const auto owners = mesh_.faceOwners().view();
    const auto neighbors = mesh_.faceNeighbors().view();

    const auto faceCenters = mesh_.faceCenters().view();
    const auto cellCenters = mesh_.cellCenters().view();
    const auto faceNormals = mesh_.faceNormals().view();

    const auto [weightS, weightB] = views(weights.internalVector(), weights.boundaryData().value());
    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            scalar sfdOwn =
                std::abs(faceNormals[facei] & (faceCenters[facei] - cellCenters[owners[facei]]));
            scalar sfdNei =
                std::abs(faceNormals[facei] & (cellCenters[neighbors[facei]] - faceCenters[facei]));

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
#ifdef NF_WITH_MPI_SUPPORT
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto surfFaceCells = mesh_.boundaryMesh().faceOwners().view();
        const auto bFaceCenters = mesh_.boundaryMesh().faceCenters().view();
        const auto bFaceNormals = mesh_.boundaryMesh().faceNormals().view();
        const auto bFaceAreas = mesh_.boundaryMesh().faceAreas().view();
        // dNei[procFacei] == |n & (Cf - Cnei)| received from the neighbouring rank
        const auto dNei = exchangeProcOwnerDistance(exec, mesh_);
        const auto dNeiView = dNei.view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                const Vec3 co = cellCenters[surfFaceCells[bfi]];
                const scalar dOwn = std::abs(n & (bFaceCenters[bfi] - co));
                const scalar dNeiF = dNeiView[procFacei];
                const scalar sum = dOwn + dNeiF;
                weightB[bfi] = (sum > ROOTVSMALL) ? dNeiF / sum : 0.5;
            },
            "basicGeometricScheme::updateWeightsProcBoundary"
        );
    }
#endif
}

void BasicGeometryScheme::updateDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& deltaCoeffs
)
{
    const auto [owners, neighbors, surfFaceCells] =
        views(mesh_.faceOwners(), mesh_.faceNeighbors(), mesh_.boundaryMesh().faceOwners());


    const auto [bFaceCenters, cellCenters] =
        views(mesh_.boundaryMesh().faceCenters(), mesh_.cellCenters());

    auto deltaCoeff = deltaCoeffs.internalVector().view();
    auto deltaCoeffB = deltaCoeffs.boundaryData().value().view();

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
            deltaCoeff[facei] = 1.0 / std::max(mag(cellToCellDist), scalar(ROOTVSMALL));
        },
        "basicGeometricScheme::updateDeltaCoeffsInternal"
    );

    parallelFor(
        exec,
        {0, mesh_.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = surfFaceCells[bfi];
            Vec3 cellToFaceDist = bFaceCenters[bfi] - cellCenters[own];
            deltaCoeffB[bfi] = 1.0 / std::max(mag(cellToFaceDist), scalar(ROOTVSMALL));
        },
        "basicGeometricScheme::updateDeltaCoeffsBoundary"
    );
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owners, neighbors, surfFaceCells] =
        views(mesh_.faceOwners(), mesh_.faceNeighbors(), mesh_.boundaryMesh().faceOwners());


    const auto [faceCenters, cellCenters, faceNormals, faceAreas] =
        views(mesh_.faceCenters(), mesh_.cellCenters(), mesh_.faceNormals(), mesh_.faceAreas());

    auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalVector().view();
    auto nonOrthDeltaCoeffB = nonOrthDeltaCoeffs.boundaryData().value().view();
    fill(nonOrthDeltaCoeffs.internalVector(), 0.0);

    const auto nInternalFaces = mesh_.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
            Vec3 faceNormal = 1 / faceAreas[facei] * faceNormals[facei];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeff[facei] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsInternal"
    );

    const auto bFaceCenters = mesh_.boundaryMesh().faceCenters().view();
    const auto bFaceNormals = mesh_.boundaryMesh().faceNormals().view();
    const auto bFaceAreas = mesh_.boundaryMesh().faceAreas().view();
    parallelFor(
        exec,
        {0, mesh_.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = surfFaceCells[bfi];
            Vec3 cellToCellDist = bFaceCenters[bfi] - cellCenters[own];
            Vec3 faceNormal = 1 / bFaceAreas[bfi] * bFaceNormals[bfi];
            scalar orthoDist = faceNormal & cellToCellDist;
            nonOrthDeltaCoeffB[bfi] = 1.0 / std::max(orthoDist, 0.05 * mag(cellToCellDist));
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );

#ifdef NF_WITH_MPI_SUPPORT
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto dNei = exchangeProcOwnerDistance(exec, mesh_);
        const auto dNeiView = dNei.view();
        const auto [bFaceNormals, bFaceArea, bFaceCenters] = views(
            mesh_.boundaryMesh().faceNormals(),
            mesh_.boundaryMesh().faceAreas(),
            mesh_.boundaryMesh().faceCenters()
        );
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceArea[bfi]) * bFaceNormals[bfi];
                const Vec3 co = cellCenters[surfFaceCells[bfi]];
                const scalar dOwn = std::abs(n & (bFaceCenters[bfi] - co));
                nonOrthDeltaCoeffB[bfi] =
                    1.0 / std::max(dOwn + dNeiView[procFacei], scalar(ROOTVSMALL));
            },
            "basicGeometricScheme::updateNonOrthDeltaCoeffsProcBoundary"
        );
    }
#endif
}


void BasicGeometryScheme::updateNonOrthCorrectionVec3s(
    const Executor& exec, SurfaceField<Vec3>& nonOrthCorrectionVec3s
)
{
    const auto [owners, neighbors] = views(mesh_.faceOwners(), mesh_.faceNeighbors());

    const auto [cellCenters, faceNormals, faceAreas] =
        views(mesh_.cellCenters(), mesh_.faceNormals(), mesh_.faceAreas());

    const auto [corrVec, corrVecB] = views(
        nonOrthCorrectionVec3s.internalVector(), nonOrthCorrectionVec3s.boundaryData().value()
    );

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 delta = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
            Vec3 n = (1.0 / faceAreas[facei]) * faceNormals[facei];
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
