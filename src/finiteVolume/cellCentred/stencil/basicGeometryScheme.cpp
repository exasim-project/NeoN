// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/distributed/haloExchange.hpp"
// cachedCommunicationPattern is declared in communicationPattern.hpp (above)
#endif

namespace NeoN::finiteVolume::cellCentred
{

// Over-relaxed non-orthogonal correction clamp factor: bounds nonOrthDeltaCoeffs
// away from a vanishing denominator on highly skewed faces.
// 0.05 is the conventional over-relaxed non-orthogonal clamp value.
constexpr scalar NON_ORTH_DELTA_CLAMP = 0.05;

#ifdef NF_WITH_MPI_SUPPORT
namespace
{

/** @brief Exchanges owner cell centres across processor boundaries: each rank sends, for every
 *  processor face, the centre of the cell owning that face, and receives the neighbouring rank's
 *  owner-cell centre. The returned device Vector (size nProcBoundaryFaces) holds, per processor
 *  face, the centre of the cell on the far side of the rank boundary (Cnei). Plain processor
 *  patches share the global coordinate system, so Cnei is directly comparable to the local owner
 *  centre. Enables exact owner-to-neighbour geometry (|Cnei - Cown|, non-orth correction) at
 *  processor faces on non-orthogonal meshes. This is the single proc-face exchange per geometry
 *  update: weights, nonOrthDeltaCoeffs and the correction vectors are all derived locally from
 *  Cnei (the neighbour face-normal distance is |n . (Cf - Cnei)|, not a separate exchange). */
Vector<Vec3> exchangeProcNeighbourCellCentre(const Executor& exec, const UnstructuredMesh& mesh)
{
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces == 0) return Vector<Vec3>(exec, 0, zero<Vec3>());

    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Gather the owner-cell centre of each processor face on the device, so that only
    // the nProcFaces Vec3s actually exchanged reach the host.
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

    // Exchange the per-proc-face owner centres through the unified halo primitive. The pattern is
    // retrieved from the mesh stencilDB cache (computed once per mesh via
    // cachedCommunicationPattern). boundaryMapVector scatters the rank-grouped recv buffer back
    // into proc-face order, so neiCentreDev[f] is the far-side owner centre (Cnei) of proc face f.
    const auto& pattern = cachedCommunicationPattern(mesh);
    Vector<Vec3> neiCentreDev(exec, nProcFaces, zero<Vec3>());
    haloExchange<Vec3>(exec, mesh, ownCentreDev.data(), neiCentreDev.data(), pattern);
    return neiCentreDev;
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
            // Both projections are intrinsically positive on a well-formed mesh (owner and
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
    // updateWeights is the first kernel GeometryScheme::update() runs, so it owns the single
    // proc-face halo exchange for the whole update: it exchanges the neighbour-owner cell centres
    // once and caches them in procNeiCellCentre_. updateNonOrthDeltaCoeffs and
    // updateNonOrthCorrectionVec3s reuse the cache, so a full geometry update performs ONE
    // exchange instead of one per kernel. The neighbour distance is derived locally from Cnei
    // (dNei = |n . (Cf - Cnei)|), so no separate scalar exchange is needed.
    procNeiCellCentre_ = exchangeProcNeighbourCellCentre(exec, mesh_);
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto surfFaceCells = mesh_.boundaryMesh().faceOwners().view();
        const auto bFaceCenters = mesh_.boundaryMesh().faceCenters().view();
        const auto bFaceNormals = mesh_.boundaryMesh().faceNormals().view();
        const auto bFaceAreas = mesh_.boundaryMesh().faceAreas().view();
        const auto cneiView = procNeiCellCentre_->view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                const Vec3 co = cellCenters[surfFaceCells[bfi]];
                const scalar dOwn = std::abs(n & (bFaceCenters[bfi] - co));
                const scalar dNeiF = std::abs(n & (bFaceCenters[bfi] - cneiView[procFacei]));
                const scalar sum = dOwn + dNeiF;
                weightB[bfi] = (sum > ROOTVSMALL) ? dNeiF / sum : 0.5;
            },
            "basicGeometricScheme::updateWeightsProcBoundary"
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
            // produce 1/0 -> inf
            nonOrthDeltaCoeff[facei] =
                1.0
                / std::max(
                    orthoDist,
                    std::max(NON_ORTH_DELTA_CLAMP * mag(cellToCellDist), scalar(ROOTVSMALL))
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
            // floor with ROOTVSMALL so a degenerate face cannot produce 1/0 -> inf
            nonOrthDeltaCoeffB[bfi] =
                1.0
                / std::max(
                    orthoDist,
                    std::max(NON_ORTH_DELTA_CLAMP * mag(cellToFaceDist), scalar(ROOTVSMALL))
                );
        },
        "basicGeometricScheme::updateNonOrthDeltaCoeffsBoundary"
    );

#ifdef NF_WITH_MPI_SUPPORT
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        // Reuse the neighbour centres exchanged in updateWeights (update() runs it first), so the
        // proc-face delta is the exact owner->neighbour vector across the rank boundary.
        NF_ASSERT(
            procNeiCellCentre_.has_value(),
            "procNeiCellCentre_ not exchanged; updateWeights must run before "
            "updateNonOrthDeltaCoeffs"
        );
        const auto cneiView = procNeiCellCentre_->view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                const Vec3 cellToCellDist = cneiView[procFacei] - cellCenters[surfFaceCells[bfi]];
                const scalar orthoDist = n & cellToCellDist;
                // Same over-relaxed clamp as the internal/boundary faces so proc faces are not a
                // special case: floor at 0.05|d| (and ROOTVSMALL) to bound a vanishing denominator.
                nonOrthDeltaCoeffB[bfi] =
                    1.0
                    / std::max(
                        orthoDist,
                        std::max(NON_ORTH_DELTA_CLAMP * mag(cellToCellDist), scalar(ROOTVSMALL))
                    );
            },
            "basicGeometricScheme::updateNonOrthDeltaCoeffsProcBoundary"
        );
    }
#endif
}


void BasicGeometryScheme::updateNonOrthCorrectionVec3s(
    const Executor& exec,
    const SurfaceField<scalar>& nonOrthDeltaCoeffs,
    SurfaceField<Vec3>& nonOrthCorrectionVec3s
)
{
    const auto [owners, neighbors] = views(mesh_.faceOwners(), mesh_.faceNeighbors());

    const auto [cellCenters, faceNormals, faceAreas] =
        views(mesh_.cellCenters(), mesh_.faceNormals(), mesh_.faceAreas());

    const auto [corrVec, corrVecB] = views(
        nonOrthCorrectionVec3s.internalVector(), nonOrthCorrectionVec3s.boundaryData().value()
    );
    // Read the precomputed nonOrthDeltaCoeff instead of re-deriving the
    // 1/max(n.d, 0.05|d|) formula: single source of truth, and the clamped/floored
    // coefficient is inherited automatically.
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
    // BoundaryData's zero-init.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { corrVecB[bfi] = zero<Vec3>(); },
        "basicGeometricScheme::updateNonOrthCorrectionVec3sBoundary"
    );

#ifdef NF_WITH_MPI_SUPPORT
    // Processor faces have a real neighbour cell across the rank boundary, so on a non-orthogonal
    // mesh the correction is non-zero there. Compute it with the same form as the
    // internal loop, using the exchanged neighbour cell centre and the precomputed processor
    // nonOrthDeltaCoeff. On an orthogonal proc face this evaluates to zero, as before.
    const auto nProcBoundaryFaces = mesh_.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        // Reuse the neighbour centres exchanged in updateWeights (update() runs it first).
        NF_ASSERT(
            procNeiCellCentre_.has_value(),
            "procNeiCellCentre_ not exchanged; updateWeights must run before "
            "updateNonOrthCorrectionVec3s"
        );
        const auto nonOrthDeltaCoeffB = nonOrthDeltaCoeffs.boundaryData().value().view();
        const auto surfFaceCells = mesh_.boundaryMesh().faceOwners().view();
        const auto [bFaceNormals, bFaceAreas] =
            views(mesh_.boundaryMesh().faceNormals(), mesh_.boundaryMesh().faceAreas());
        const auto cneiView = procNeiCellCentre_->view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const localIdx bfi = nBoundaryFaces + procFacei;
                const Vec3 n = (1.0 / bFaceAreas[bfi]) * bFaceNormals[bfi];
                const Vec3 delta = cneiView[procFacei] - cellCenters[surfFaceCells[bfi]];
                corrVecB[bfi] = n - delta * nonOrthDeltaCoeffB[bfi];
            },
            "basicGeometricScheme::updateNonOrthCorrectionVec3sProcBoundary"
        );
    }
#endif
}

} // namespace NeoN
