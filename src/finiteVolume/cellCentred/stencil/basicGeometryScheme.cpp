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
// 0.05 is the conventional over-relaxed non-orthogonal clamp value.
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
    mpi::waitAll(requests);

    // M6: allocate the result directly on exec from the host buffer (no SerialExecutor detour)
    return Vector<scalar>(exec, dNei);
}

// Tag for the geometry-scheme processor neighbour-cell-centre halo exchange (review v2a).
constexpr mpi_label_t procNeighbourCentreTag = 0x6763; // 'gc'

/** @brief Exchanges owner cell centres across processor boundaries: each rank sends, for every
 *  processor face, the centre of the cell owning that face, and receives the neighbouring rank's
 *  owner-cell centre. The returned device Vector (size nProcBoundaryFaces) holds, per processor
 *  face, the centre of the cell on the far side of the rank boundary (Cnei). Plain processor
 *  patches share the global coordinate system, so Cnei is directly comparable to the local owner
 *  centre. Enables exact owner-to-neighbour geometry (|Cnei - Cown|, non-orth correction) at
 *  processor faces on non-orthogonal meshes. */
Vector<Vec3> exchangeProcNeighbourCellCentre(const Executor& exec, const UnstructuredMesh& mesh)
{
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces == 0) return Vector<Vec3>(exec, 0, zero<Vec3>());

    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Gather the owner-cell centre of each processor face on the device (H4: only nProcFaces
    // Vec3s reach the host).
    Vector<Vec3> ownCentreDev(exec, nProcFaces, zero<Vec3>());
    {
        auto ownView = ownCentreDev.view();
        const auto cellCenters = mesh.cellCenters().view();
        const auto bFaceOwners = bMesh.faceOwners().view();
        parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const localIdx i) {
                ownView[i] = cellCenters[bFaceOwners[nBoundaryFaces + i]];
            },
            "basicGeometricScheme::exchangeProcNeighbourCellCentreOwner"
        );
    }

    auto ownH = ownCentreDev.copyToHost();
    const auto ownHView = ownH.view();
    // Flatten to contiguous [x, y, z] scalars per face for the typed MPI exchange.
    std::vector<scalar> sendBuf(static_cast<std::size_t>(3 * nProcFaces));
    for (localIdx i = 0; i < nProcFaces; ++i)
    {
        const Vec3 c = ownHView[i];
        sendBuf[static_cast<std::size_t>(3 * i + 0)] = c[0];
        sendBuf[static_cast<std::size_t>(3 * i + 1)] = c[1];
        sendBuf[static_cast<std::size_t>(3 * i + 2)] = c[2];
    }
    std::vector<scalar> recvBuf(static_cast<std::size_t>(3 * nProcFaces), scalar(0));

    const auto ranges = collectProcPatchRanges(mesh);
    std::vector<MPI_Request> requests(2 * ranges.size(), MPI_REQUEST_NULL);
    mpi::Environment mpiEnv;
    for (std::size_t p = 0; p < ranges.size(); ++p)
    {
        const auto [rangeStart, rangeEnd] = ranges[p];
        const localIdx patchOff = 3 * (rangeStart - nBoundaryFaces);
        const auto neighborRank = static_cast<mpi_label_t>(bMesh.neighbourRankForRange(ranges[p]));
        const auto count = static_cast<mpi_label_t>(3 * (rangeEnd - rangeStart));
        mpi::isend<scalar>(
            sendBuf.data() + patchOff,
            count,
            neighborRank,
            procNeighbourCentreTag,
            mpiEnv.comm(),
            &requests[2 * p]
        );
        mpi::irecv<scalar>(
            recvBuf.data() + patchOff,
            count,
            neighborRank,
            procNeighbourCentreTag,
            mpiEnv.comm(),
            &requests[2 * p + 1]
        );
    }
    mpi::waitAll(requests);

    std::vector<Vec3> neiCentre(static_cast<std::size_t>(nProcFaces));
    for (localIdx i = 0; i < nProcFaces; ++i)
        neiCentre[static_cast<std::size_t>(i)] = Vec3 {
            recvBuf[static_cast<std::size_t>(3 * i + 0)],
            recvBuf[static_cast<std::size_t>(3 * i + 1)],
            recvBuf[static_cast<std::size_t>(3 * i + 2)]
        };
    return Vector<Vec3>(exec, neiCentre);
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
        // Exact processor-boundary deltaCoeffs (review v2a / GEOM-03): the orthogonal
        // deltaCoeffs is 1/|Cnei - Cown| across the rank boundary (the coupled-patch
        // deltaCoeffs). Exchanging the neighbour cell centre makes this exact on
        // non-orthogonal processor faces too (previously a face-normal projection was used).
        const auto Cnei = exchangeProcNeighbourCellCentre(exec, mesh_);
        const auto CneiView = Cnei.view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 delta = CneiView[procFacei] - cellCenters[surfFaceCells[bfi]];
                deltaCoeffB[bfi] = 1.0 / std::max(mag(delta), scalar(ROOTVSMALL));
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

    // Non-processor patches are one-sided, so corrVec is zero there (the snGrad reduces to the
    // uncorrected form). Zero them explicitly so consumers may rely on the contract rather than on
    // BoundaryData's zero-init (review N6).
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { corrVecB[bfi] = zero<Vec3>(); },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sBoundary"
    );

#ifdef NF_WITH_MPI_SUPPORT
    // Processor faces have a real neighbour cell across the rank boundary, so on a non-orthogonal
    // mesh the correction is non-zero there (review v2a / N4). Compute it with the same form as the
    // internal loop, using the exchanged neighbour cell centre and the precomputed processor
    // nonOrthDeltaCoeff. On an orthogonal proc face this evaluates to zero, as before.
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto nonOrthDeltaCoeffB = nonOrthDeltaCoeffs.boundaryData().value().view();
        const auto surfFaceCells = mesh_.boundaryMesh().faceOwners().view();
        const auto [bFaceNormals, bFaceAreas] =
            views(mesh_.boundaryMesh().faceNormals(), mesh_.boundaryMesh().faceAreas());
        const auto Cnei = exchangeProcNeighbourCellCentre(exec, mesh_);
        const auto CneiView = Cnei.view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                const Vec3 delta = CneiView[procFacei] - cellCenters[surfFaceCells[bfi]];
                corrVecB[bfi] = n - delta * nonOrthDeltaCoeffB[bfi];
            },
            "basicGeometricScheme::updateNonOrthCorrectionVec3sProcBoundary"
        );
    }
#endif
}

} // namespace NeoN
