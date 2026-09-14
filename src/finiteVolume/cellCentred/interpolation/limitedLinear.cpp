// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/interpolation/limitedLinear.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

namespace
{

NEON_INLINE_FUNCTION
scalar signOf(const scalar s) { return s >= scalar(0) ? scalar(1) : scalar(-1); }

/* @brief TVD ratio r of the upwind-cell gradient to the face gradient.
 *
 * A face gradient negligible against the cell gradient carries no usable slope information, so
 * the ratio saturates at +/-2000 rather than dividing. The divisor of the regular branch is
 * floored rather than merely branch-guarded: the compiler if-converts the two branches into
 * unconditional divisions, so a zero face gradient would evaluate 0/0 and raise FE_INVALID when
 * floating-point traps are enabled, even though the quotient is discarded.
 */
NEON_INLINE_FUNCTION
scalar tvdRatio(const scalar gradf, const scalar gradcf)
{
    if (Kokkos::abs(gradcf) >= scalar(1000) * Kokkos::abs(gradf))
    {
        return scalar(2000) * signOf(gradcf) * signOf(gradf) - scalar(1);
    }
    // The floor keeps the magnitude away from zero but must keep gradf's SIGN: clamping a tiny
    // negative face gradient to +ROOTVSMALL would flip the sign of r and force upwind on a face
    // where both gradients agree in sign and central differencing is correct.
    const scalar denom = Kokkos::abs(gradf) > ROOTVSMALL ? gradf : signOf(gradf) * ROOTVSMALL;
    return scalar(2) * (gradcf / denom) - scalar(1);
}

NEON_INLINE_FUNCTION
scalar limitedLinearLimiter(const scalar twoByk, const scalar gradf, const scalar gradcf)
{
    return Kokkos::max(Kokkos::min(twoByk * tvdRatio(gradf, gradcf), scalar(1)), scalar(0));
}

/* @brief owner-to-neighbour displacement C_nei - C_own of every processor face, cached per mesh.
 *
 * BoundaryMesh::delta() only stores Cf - C_local, which is half of the centre-to-centre vector the
 * TVD ratio needs; using it directly halves gradcf and can select a different limiter for no reason
 * other than the face crossing a rank boundary. The missing half is the neighbouring rank's own
 * delta: with delta_x = Cf - C_x, C_nei - C_own = delta_own - delta_nei. This is pure geometry, so
 * the halo exchange runs once per mesh and every later weight evaluation reads the cached vector.
 */
std::shared_ptr<Vector<Vec3>> procFaceDelta(const Executor& exec, const UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    const std::string key = "limitedLinear::procFaceDelta";
    if (db.contains(key))
    {
        return db.get<std::shared_ptr<Vector<Vec3>>>(key);
    }

    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();

    // This field is only the transport for the exchange: its proc-face slots are filled with the
    // local delta, and communicate() replaces them with the neighbouring rank's delta.
    SurfaceField<Vec3> remoteDelta(
        exec, "limitedLinearProcDelta", mesh, createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
    );
    auto remoteV = remoteDelta.boundaryData().value().view();
    const auto bDelta = bMesh.delta().view();

    auto delta = std::make_shared<Vector<Vec3>>(exec, nProcBoundaryFaces, zero<Vec3>());
    auto deltaV = delta->view();

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            const auto bcfacei = nBoundaryFaces + procFacei;
            remoteV[bcfacei] = bDelta[bcfacei];
        },
        "limitedLinearProcDeltaFill"
    );

#ifdef NF_WITH_MPI_SUPPORT
    fence(exec);
    const auto nBounds = bMesh.nBoundaries();
    const auto nProcPatches = bMesh.nProcBoundaryPatches();
    for (localIdx patchi = nBounds - nProcPatches; patchi < nBounds; ++patchi)
    {
        const auto range = remoteDelta.boundaryData().range(patchi);
        const auto neighborRank = static_cast<int>(bMesh.neighbourRankForRange(range));
        remoteDelta.boundaryData().communicate(range, neighborRank);
    }
    remoteDelta.boundaryData().waitAll();
#endif

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            const auto bcfacei = nBoundaryFaces + procFacei;
            deltaV[procFacei] = bDelta[bcfacei] - remoteV[bcfacei];
        },
        "limitedLinearProcDeltaCombine"
    );

    db.insert(key, delta);
    return delta;
}

} // namespace

void computeLimitedLinearWeights(
    const VolumeField<scalar>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& cdWeights,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const scalar twoByk,
    SurfaceField<scalar>& weights
)
{
    const auto exec = weights.exec();
    const auto& mesh = weights.mesh();

    // Processor-aware calculated BCs so correctBoundaryConditions() fills the proc-face tail with
    // the NEIGHBOUR cell gradient, which the limiter needs when the neighbour is upwind.
    VolumeField<Vec3> gradPhi(
        exec, "limitedLinearGrad", mesh, createCalculatedProcBCs<VolumeBoundary<Vec3>>(mesh)
    );
    fill(gradPhi.internalVector(), zero<Vec3>());
    fill(gradPhi.boundaryData().value(), zero<Vec3>());
    GaussGreenGrad(exec, mesh).grad(src, gradPhi);
    gradPhi.correctBoundaryConditions();

    auto wS = weights.internalVector().view();
    auto wB = weights.boundaryData().value().view();
    const auto [srcS, gradS, cdS, dOwnS, dNeiS, fluxS, ownerS, neighS] = views(
        src.internalVector(),
        gradPhi.internalVector(),
        cdWeights.internalVector(),
        faceDeltaOwner.internalVector(),
        faceDeltaNeighbour.internalVector(),
        flux.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors()
    );

    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = ownerS[facei];
            const auto nei = neighS[facei];
            // d points owner -> neighbour: (Cf - C_own) - (Cf - C_nei).
            const Vec3 d = dOwnS[facei] - dNeiS[facei];
            const scalar gradf = srcS[nei] - srcS[own];
            const scalar gradcf = d & (fluxS[facei] > scalar(0) ? gradS[own] : gradS[nei]);
            const scalar limiter = limitedLinearLimiter(twoByk, gradf, gradcf);
            const scalar upwindW = fluxS[facei] >= scalar(0) ? scalar(1) : scalar(0);
            wS[facei] = limiter * cdS[facei] + (scalar(1) - limiter) * upwindW;
        },
        "computeLimitedLinearWeightsInternal"
    );

    // Physical boundary faces take the patch value, as for the linear and upwind schemes.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { wB[bfi] = scalar(1); },
        "computeLimitedLinearWeightsBoundary"
    );

    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto& bMesh = mesh.boundaryMesh();
        const auto bOwners = bMesh.faceOwners().view();
        // normalSign (+1 owner-side, -1 non-owner-side) of the rank-local cell, stored in the
        // boundary weights for processor patches; see linearUpwind's correction path.
        const auto bNormalSignV = bMesh.weights().view();
        const auto bSrcV = src.boundaryData().value().view();
        const auto bGradV = gradPhi.boundaryData().value().view();
        const auto bFluxV = flux.boundaryData().value().view();
        const auto bCdV = cdWeights.boundaryData().value().view();
        // Full C_nei - C_own for the proc faces, oriented from the rank-local cell outwards.
        const auto procDelta = procFaceDelta(exec, mesh);
        const auto dProcV = procDelta->view();

        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bcfacei = nBoundaryFaces + procFacei;
                const auto own = bOwners[bcfacei];
                // bOwners is the rank-local cell, but the stored proc-face flux is in the global
                // face orientation, so the flux LEAVING the local cell is normalSign * F. Both the
                // upwind-cell selection and the local-cell weight must use that; the raw flux makes
                // the non-owner rank pick the opposite upwind cell and write the opposite weight.
                const scalar outwardFlux = bNormalSignV[bcfacei] * bFluxV[bcfacei];
                const scalar gradf = bSrcV[bcfacei] - srcS[own];
                const scalar gradcf =
                    dProcV[procFacei] & (outwardFlux > scalar(0) ? gradS[own] : bGradV[bcfacei]);
                const scalar limiter = limitedLinearLimiter(twoByk, gradf, gradcf);
                const scalar upwindW = outwardFlux >= scalar(0) ? scalar(1) : scalar(0);
                wB[bcfacei] = limiter * bCdV[bcfacei] + (scalar(1) - limiter) * upwindW;
            },
            "computeLimitedLinearWeightsProcBoundary"
        );
    }
}

} // namespace NeoN::finiteVolume::cellCentred
