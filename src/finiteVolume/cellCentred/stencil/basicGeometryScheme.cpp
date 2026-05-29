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

// Over-relaxed non-orthogonal correction clamp factor (review L1): bounds
// nonOrthDeltaCoeffs away from a vanishing denominator on highly skewed faces.
// Matches OpenFOAM's surfaceInterpolation default.
constexpr scalar nonOrthDeltaClamp = 0.05;

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
// Tag for the geometry-scheme processor-owner-distance halo exchange. Distinct from
// BoundaryData::communicate (tag 0) so the two cannot mismatch if they ever overlap on
// the same rank pair (review M5).
constexpr mpi_label_t procOwnerDistanceTag = 0x6764; // 'gd'

Vector<scalar> exchangeProcOwnerDistance(const Executor& exec, const UnstructuredMesh& mesh)
{
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    // L7: nothing to exchange on a serial / interior-only partition
    if (nProcFaces == 0) return Vector<scalar>(exec, 0, scalar(0));

    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    // H4: compute the owner projected distance for each processor face on the device,
    // reading only device-resident geometry. This avoids the full mesh.cellCenters() D->H
    // copy (GB-scale on an industrial mesh) plus the four boundary-array host copies — only
    // the nProcFaces scalars actually exchanged are moved to the host.
    Vector<scalar> dOwnDev(exec, nProcFaces, scalar(0));
    {
        auto dOwnView = dOwnDev.view();
        const auto cellCenters = mesh.cellCenters().view();
        const auto bFaceCenters = bMesh.faceCenters().view();
        const auto bFaceNormals = bMesh.faceNormals().view();
        const auto bFaceAreas = bMesh.faceAreas().view();
        const auto bFaceOwners = bMesh.faceOwners().view();
        parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const localIdx i) {
                const localIdx bfi = nBoundaryFaces + i;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                dOwnView[i] = std::abs(n & (bFaceCenters[bfi] - cellCenters[bFaceOwners[bfi]]));
            },
            "basicGeometricScheme::exchangeProcOwnerDistanceOwner"
        );
    }

    auto dOwnH = dOwnDev.copyToHost();
    const auto dOwnHView = dOwnH.view();
    std::vector<scalar> dOwn(static_cast<std::size_t>(nProcFaces));
    for (localIdx i = 0; i < nProcFaces; ++i)
        dOwn[static_cast<std::size_t>(i)] = dOwnHView[i];
    std::vector<scalar> dNei(static_cast<std::size_t>(nProcFaces), scalar(0));

    const auto ranges = collectProcPatchRanges(mesh);
    std::vector<MPI_Request> requests(2 * ranges.size(), MPI_REQUEST_NULL);
    mpi::Environment mpiEnv;
    for (std::size_t p = 0; p < ranges.size(); ++p)
    {
        const auto [rangeStart, rangeEnd] = ranges[p];
        const localIdx patchOff = rangeStart - nBoundaryFaces;
        const auto neighborRank = static_cast<mpi_label_t>(bMesh.neighbourRankForRange(ranges[p]));
        const auto count = static_cast<mpi_label_t>(rangeEnd - rangeStart);
        // M5: typed scalar send/recv (MPI selects the datatype) with a meaningful tag
        mpi::isend<scalar>(
            dOwn.data() + patchOff,
            count,
            neighborRank,
            procOwnerDistanceTag,
            mpiEnv.comm(),
            &requests[2 * p]
        );
        mpi::irecv<scalar>(
            dNei.data() + patchOff,
            count,
            neighborRank,
            procOwnerDistanceTag,
            mpiEnv.comm(),
            &requests[2 * p + 1]
        );
    }
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // M6: allocate the result directly on exec from the host buffer (no SerialExecutor detour)
    return Vector<scalar>(exec, dNei);
}

} // anonymous namespace
#endif

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(), mesh_(mesh)
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
            // M8: both projections are intrinsically positive on a well-formed mesh (owner and
            // neighbour centres sit on opposite sides of the face along its normal). std::abs is
            // a deliberate robustness guard against a locally flipped/strongly-twisted face
            // yielding a negative projection and hence a nonsensical weight; the alternative
            // (failing loud) is impractical inside a device kernel.
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

void BasicGeometryScheme::updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs)
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
            // Boundary deltaCoeffs is the one-sided cell-centre-to-face-centre inverse
            // distance (no neighbour cell across a physical patch).
            // TODO(#515): revisit when the boundary delta convention is unified.
            Vec3 cellToFaceDist = bFaceCenters[bfi] - cellCenters[own];
            deltaCoeffB[bfi] = 1.0 / std::max(mag(cellToFaceDist), scalar(ROOTVSMALL));
        },
        "basicGeometricScheme::updateDeltaCoeffsBoundary"
    );

#ifdef NF_WITH_MPI_SUPPORT
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        // Processor-boundary deltaCoeffs (review GEOM-03 / H1-remainder). Uses the
        // face-normal-projected owner+neighbour distance (dOwn + dNei). On an
        // orthogonal proc face this equals 1/|d| (OF's coupled deltaCoeffs); on a
        // non-orthogonal proc face it coincides with nonOrthDeltaCoeffs instead of
        // OF's euclidean 1/|d|. Recovering exact 1/|d| there needs the full
        // neighbour cell centre exchanged (deferred). Leaving these at zero — the
        // pre-fix behaviour — is strictly worse (zero diffusive flux across ranks).
        const auto dNei = exchangeProcOwnerDistance(exec, mesh_);
        const auto dNeiView = dNei.view();
        const auto [bFaceNormals, bFaceArea] =
            views(mesh_.boundaryMesh().faceNormals(), mesh_.boundaryMesh().faceAreas());
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceArea[bfi]) * bFaceNormals[bfi];
                const scalar dOwn =
                    std::abs(n & (bFaceCenters[bfi] - cellCenters[surfFaceCells[bfi]]));
                deltaCoeffB[bfi] = 1.0 / std::max(dOwn + dNeiView[procFacei], scalar(ROOTVSMALL));
            },
            "basicGeometricScheme::updateDeltaCoeffsProcBoundary"
        );
    }
#endif
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owners, neighbors, surfFaceCells] =
        views(mesh_.faceOwners(), mesh_.faceNeighbors(), mesh_.boundaryMesh().faceOwners());


    // Internal faces only (post boundaryMesh dedup): mesh_.faceNormals()/faceAreas() are sized to
    // the internal-face count; boundary-face geometry comes from the boundary mesh views below.
    const auto [cellCenters, faceNormals, faceAreas] =
        views(mesh_.cellCenters(), mesh_.faceNormals(), mesh_.faceAreas());

    auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalVector().view();
    auto nonOrthDeltaCoeffB = nonOrthDeltaCoeffs.boundaryData().value().view();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 cellToCellDist = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
            Vec3 faceUnitNormal = 1 / faceAreas[facei] * faceNormals[facei];
            scalar orthoDist = faceUnitNormal & cellToCellDist;
            // floor with ROOTVSMALL so a degenerate (coincident-centre) face cannot
            // produce 1/0 -> inf (review H3)
            nonOrthDeltaCoeff[facei] =
                1.0
                / std::max(
                    orthoDist, std::max(nonOrthDeltaClamp * mag(cellToCellDist), scalar(ROOTVSMALL))
                );
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
            Vec3 cellToFaceDist = bFaceCenters[bfi] - cellCenters[own];
            Vec3 faceNormal = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
            scalar orthoDist = faceNormal & cellToFaceDist;
            // floor with ROOTVSMALL (review H3)
            nonOrthDeltaCoeffB[bfi] =
                1.0
                / std::max(
                    orthoDist, std::max(nonOrthDeltaClamp * mag(cellToFaceDist), scalar(ROOTVSMALL))
                );
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
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
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
    const Executor& exec,
    SurfaceField<Vec3>& nonOrthCorrectionVec3s,
    const SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    const auto [owners, neighbors] = views(mesh_.faceOwners(), mesh_.faceNeighbors());

    const auto [cellCenters, faceNormals, faceAreas] =
        views(mesh_.cellCenters(), mesh_.faceNormals(), mesh_.faceAreas());

    const auto [corrVec, corrVecB] = views(
        nonOrthCorrectionVec3s.internalVector(), nonOrthCorrectionVec3s.boundaryData().value()
    );
    // read the precomputed nonOrthDeltaCoeff instead of re-deriving the
    // 1/max(n.d, 0.05|d|) formula (review M3 — single source of truth; also
    // inherits the ROOTVSMALL floor added for H3)
    const auto nonOrthDeltaCoeff = nonOrthDeltaCoeffs.internalVector().view();

    const auto nInternalFaces = mesh_.nInternalFaces();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Vec3 delta = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
            Vec3 n = (1.0 / faceAreas[facei]) * faceNormals[facei];
            corrVec[facei] = n - delta * nonOrthDeltaCoeff[facei];
        },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sInternal"
    );

    // corrVec is zero on ALL boundary faces: non-processor patches are one-sided so
    // the snGrad reduces to the uncorrected form, and the non-orthogonal correction
    // at processor faces is currently deferred (review N4). Zero both the non-proc
    // and proc ranges explicitly so consumers may rely on the contract rather than
    // on BoundaryData's zero-init (review N6).
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    parallelFor(
        exec,
        {0, nBoundaryFaces + nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { corrVecB[bfi] = zero<Vec3>(); },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sBoundary"
    );
}

} // namespace NeoN
