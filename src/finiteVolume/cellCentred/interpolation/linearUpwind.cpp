// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linearUpwind.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

namespace
{
// Type-generic "dot product" used only by the OpenFOAM-style vector clip below: for Vec3 this is
// the usual inner product; for scalar it degenerates to plain multiplication so the clip's
// formula (which is only ever runtime-enabled for Vec3) still compiles for the scalar
// instantiation of applyLinearUpwind. Written this way -- rather than branching on
// `if constexpr (std::is_same_v<ValueType, Vec3>)` inside the lambda -- because nvcc forbids
// first-capturing a variable inside an `if constexpr` block in an extended host/device lambda
// (see applyLinearUpwind's own withSrcValue comment below for the same constraint elsewhere in
// this file).
NEON_INLINE_FUNCTION scalar luwDot(const scalar a, const scalar b) { return a * b; }
NEON_INLINE_FUNCTION scalar luwDot(const Vec3& a, const Vec3& b) { return a & b; }
} // namespace

/* @brief applies the linearUpwind reconstruction for every face.
**
** @tparam ValueType field value type (scalar or Vec3)
** @tparam GradType cell gradient type (Vec3 for scalar fields, Tensor for Vec3 fields)
** @param weights linear-interpolation owner-side weight per internal face; only read when
**        ofStyleVectorClip is set.
** @param withSrcValue when true dst is the full face value (upwind value + correction); when false
**        dst holds only the gradient correction (used for the implicit deferred correction). A
**        runtime flag is used rather than a template/`if constexpr`, since nvcc forbids
**        first-capturing a variable inside an `if constexpr` block in an extended device lambda.
** @param ofStyleVectorClip "linearUpwindV" (Vec3 fields only; ignored for scalar): after computing
**        the unlimited correction, clip it exactly as OpenFOAM's linearUpwindV::correction() does
**        (linearUpwindV.C:90-125) -- bound the correction vector, as a whole, by that one face's
**        own upwind/downwind value jump:
**          maxCorr = ownerUpwind ? (1-w)*(U_nei - U_own) : w*(U_own - U_nei)
**          if (corr . corr > 0):
**            if (corr . maxCorr < 0)        corr = 0
**            elif (corr . corr > corr . maxCorr)  corr *= (corr . maxCorr) / (corr . corr)
**        This differs from a cell-wide minmod-limited gradient (which an earlier version of this
**        file used): OpenFOAM limits the per-face correction, face by face, against only that
**        face's own neighbour; it never pre-limits the gradient itself, and the three components
**        of the correction are scaled together, not independently.
**
** For both ranks the unclipped correction is `grad & d`: with NeoN's row-major tensor convention
** (gradTensor(i,j) = d U_i / d x_j) the matrix-vector product `Tensor & Vec3` reproduces
** OpenFOAM's `(Cf - C) & gradVf`; for scalars it degenerates to the Vec3 inner product.
*/
template<typename ValueType, typename GradType>
void applyLinearUpwind(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    const VolumeField<GradType>& gradPhi,
    SurfaceField<ValueType>& dst,
    const bool withSrcValue,
    const bool ofStyleVectorClip
)
{
    const auto exec = dst.exec();
    const auto& mesh = dst.mesh();

    auto dstS = dst.internalVector().view();
    auto dstB = dst.boundaryData().value().view();
    const auto [srcS, gradS, dOwnS, dNeiS, fluxS, weightsS, ownerS, neighS, boundS] = views(
        src.internalVector(),
        gradPhi.internalVector(),
        faceDeltaOwner.internalVector(),
        faceDeltaNeighbour.internalVector(),
        flux.internalVector(),
        weights.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        src.boundaryData().value()
    );

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // Upwind cell follows the flux direction; S_f points owner -> neighbour.
            const bool ownerUpwind = fluxS[facei] >= 0;
            const auto own = ownerS[facei];
            const auto nei = neighS[facei];
            const auto cell = ownerUpwind ? own : nei;
            const Vec3 d = ownerUpwind ? dOwnS[facei] : dNeiS[facei];
            ValueType corr = gradS[cell] & d;

            if (ofStyleVectorClip)
            {
                const scalar w = weightsS[facei];
                const ValueType maxCorr = ownerUpwind ? (srcS[nei] - srcS[own]) * (scalar(1) - w)
                                                      : (srcS[own] - srcS[nei]) * w;
                const scalar corrSq = luwDot(corr, corr);
                if (corrSq > scalar(0))
                {
                    const scalar maxCorrDot = luwDot(corr, maxCorr);
                    if (maxCorrDot < scalar(0))
                    {
                        corr = zero<ValueType>();
                    }
                    else if (corrSq > maxCorrDot)
                    {
                        corr = corr * (maxCorrDot / (corrSq + ROOTVSMALL));
                    }
                }
            }

            dstS[facei] = withSrcValue ? (srcS[cell] + corr) : corr;
        },
        "computeLinearUpwindInternal"
    );

    // Physical (non-coupled) boundary faces take the patch value with no correction, matching
    // OpenFOAM's linearUpwind which only corrects coupled patches; the correction-only path is
    // zero.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            dstB[bfi] = withSrcValue ? boundS[bfi] : zero<ValueType>();
        },
        "computeLinearUpwindBoundary"
    );

    // Processor (coupled) boundary faces.
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto& bMesh = mesh.boundaryMesh();
        const auto bfOwners = bMesh.faceOwners().view();
        const auto bFluxV = flux.boundaryData().value().view();

        if (withSrcValue)
        {
            // interpolate(): proc faces use the plain upwind value (the explicit Gauss div does
            // not integrate proc faces, so a deferred correction here would be unused).
            parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    const auto bcfacei = nBoundaryFaces + procFacei;
                    const auto own = bfOwners[bcfacei];
                    dstB[bcfacei] = bFluxV[bcfacei] >= 0 ? srcS[own] : boundS[bcfacei];
                },
                "computeLinearUpwindProcBoundaryUpwind"
            );
        }
        else
        {
            // correction(): a proc face is a cut internal face. Whichever rank owns the upwind
            // cell can compute its correction locally as grad[own] & (Cf - C[own]); the neighbour
            // rank's owner-side correction is exactly this face's neighbour-side correction. So
            // each rank computes its owner-side correction, exchanges it, and selects owner- vs
            // neighbour-side by the local flux sign. delta() is Cf - C[own] for boundary faces.
            // NOTE: ofStyleVectorClip is intentionally not applied here yet -- OpenFOAM's own
            // linearUpwindV::correction() does clip coupled-patch faces (linearUpwindV.C:131-192),
            // using the patch neighbour field's value for the jump term, but this investigation's
            // test case is single-rank and never exercises this path; left as a known gap rather
            // than an untested implementation.
            const auto bDeltaV = bMesh.delta().view();
            // normalSign (+1 owner-side, -1 non-owner-side): the stored proc-face flux is in the
            // patch normal orientation, so the flux LEAVING the local owner is normalSign * F.
            const auto bNormalSignV = bMesh.weights().view();
            Vector<ValueType> corrOwn(exec, nProcBoundaryFaces, zero<ValueType>());
            auto corrOwnV = corrOwn.view();
            parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    const auto bcfacei = nBoundaryFaces + procFacei;
                    const ValueType c = gradS[bfOwners[bcfacei]] & bDeltaV[bcfacei];
                    corrOwnV[procFacei] = c;
                    dstB[bcfacei] = c;
                },
                "computeLinearUpwindProcCorrectionOwner"
            );

#ifdef NF_WITH_MPI_SUPPORT
            // exchange owner-side corrections: each proc slot receives the neighbour rank's value.
            fence(exec);
            const auto nBounds = bMesh.nBoundaries();
            const auto nProcPatches = bMesh.nProcBoundaryPatches();
            for (localIdx patchi = nBounds - nProcPatches; patchi < nBounds; ++patchi)
            {
                const auto range = dst.boundaryData().range(patchi);
                const auto neighborRank = static_cast<int>(bMesh.neighbourRankForRange(range));
                dst.boundaryData().communicate(range, neighborRank);
            }
            // communicate() is asynchronous; waitAll() completes the isend/irecv and writes the
            // received neighbour values into the proc slots, replacing the local owner-side values.
            dst.boundaryData().waitAll();
#endif

            parallelFor(
                exec,
                {0, nProcBoundaryFaces},
                NEON_LAMBDA(const localIdx procFacei) {
                    const auto bcfacei = nBoundaryFaces + procFacei;
                    // owner is upwind when the flux leaves it: normalSign * F >= 0. dstB now holds
                    // the neighbour's owner-side correction (corrNei) from the exchange.
                    const scalar outwardFlux = bNormalSignV[bcfacei] * bFluxV[bcfacei];
                    dstB[bcfacei] = outwardFlux >= 0 ? corrOwnV[procFacei] : dstB[bcfacei];
                },
                "computeLinearUpwindProcCorrectionSelect"
            );
        }
    }
}

template<typename ValueType>
void computeLinearUpwind(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const bool withSrcValue,
    const bool ofStyleVectorClip
)
{
    using GradType = typename detail::LinearUpwindGradType<ValueType>::type;

    const auto exec = src.exec();
    const auto& mesh = src.mesh();

    // Both plain linearUpwind and linearUpwindV reconstruct from the same *unlimited* Gauss-Green
    // gradient (confirmed directly against OpenFOAM's linearUpwindV.C: it looks up whatever
    // gradScheme the case names, which defaults to "Gauss linear" unless a case explicitly
    // configures otherwise, and never wraps it in a cell-limiter itself). linearUpwindV's bound is
    // applied afterward, per face, in applyLinearUpwind via ofStyleVectorClip -- see that
    // function's doc comment for the exact OpenFOAM-matching formula.
    if constexpr (std::is_same_v<ValueType, scalar>)
    {
        GaussGreenGrad gradOp(exec, mesh);
        const VolumeField<GradType> gradPhi = gradOp.grad(src);
        applyLinearUpwind<ValueType, GradType>(
            src,
            flux,
            faceDeltaOwner,
            faceDeltaNeighbour,
            weights,
            gradPhi,
            dst,
            withSrcValue,
            ofStyleVectorClip
        );
    }
    else
    {
        GaussGreenGrad gradOp(exec, mesh);
        const VolumeField<GradType> gradPhi = gradOp.gradTensor(src);
        applyLinearUpwind<ValueType, GradType>(
            src,
            flux,
            faceDeltaOwner,
            faceDeltaNeighbour,
            weights,
            gradPhi,
            dst,
            withSrcValue,
            ofStyleVectorClip
        );
    }
}

template<typename ValueType>
void computeLinearUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const bool ofStyleVectorClip
)
{
    computeLinearUpwind<ValueType>(
        src, flux, faceDeltaOwner, faceDeltaNeighbour, weights, dst, true, ofStyleVectorClip
    );
}

template<typename ValueType>
void computeLinearUpwindCorrection(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const bool ofStyleVectorClip
)
{
    computeLinearUpwind<ValueType>(
        src, flux, faceDeltaOwner, faceDeltaNeighbour, weights, dst, false, ofStyleVectorClip
    );
}

#define NF_DECLARE_COMPUTE_IMP_LINUPW_INT(TYPENAME)                                                \
    template void computeLinearUpwindInterpolation<TYPENAME>(                                      \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<Vec3>&,                                                                 \
        const SurfaceField<Vec3>&,                                                                 \
        const SurfaceField<scalar>&,                                                               \
        SurfaceField<TYPENAME>&,                                                                   \
        const bool                                                                                 \
    );                                                                                             \
    template void computeLinearUpwindCorrection<TYPENAME>(                                         \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<Vec3>&,                                                                 \
        const SurfaceField<Vec3>&,                                                                 \
        const SurfaceField<scalar>&,                                                               \
        SurfaceField<TYPENAME>&,                                                                   \
        const bool                                                                                 \
    )

NF_DECLARE_COMPUTE_IMP_LINUPW_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LINUPW_INT(Vec3);

} // namespace NeoN::finiteVolume::cellCentred
