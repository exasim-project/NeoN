// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linearUpwind.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief applies the linearUpwind reconstruction for every face.
**
** @tparam ValueType field value type (scalar or Vec3)
** @tparam GradType cell gradient type (Vec3 for scalar fields, Tensor for Vec3 fields)
** @param withSrcValue when true dst is the full face value (upwind value + correction); when false
**        dst holds only the gradient correction (used for the implicit deferred correction). A
**        runtime flag is used rather than a template/`if constexpr`, since nvcc forbids
**        first-capturing a variable inside an `if constexpr` block in an extended device lambda.
**
** For both ranks the correction is `grad & d`: with NeoN's row-major tensor convention
** (gradTensor(i,j) = d U_i / d x_j) the matrix-vector product `Tensor & Vec3` reproduces
** OpenFOAM's `(Cf - C) & gradVf`; for scalars it degenerates to the Vec3 inner product.
*/
template<typename ValueType, typename GradType>
void applyLinearUpwind(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const VolumeField<GradType>& gradPhi,
    SurfaceField<ValueType>& dst,
    const bool withSrcValue
)
{
    const auto exec = dst.exec();
    const auto& mesh = dst.mesh();

    auto dstS = dst.internalVector().view();
    auto dstB = dst.boundaryData().value().view();
    const auto [srcS, gradS, dOwnS, dNeiS, fluxS, ownerS, neighS, boundS] = views(
        src.internalVector(),
        gradPhi.internalVector(),
        faceDeltaOwner.internalVector(),
        faceDeltaNeighbour.internalVector(),
        flux.internalVector(),
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
            const auto cell = ownerUpwind ? ownerS[facei] : neighS[facei];
            const Vec3 d = ownerUpwind ? dOwnS[facei] : dNeiS[facei];
            const ValueType corr = gradS[cell] & d;
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
    SurfaceField<ValueType>& dst,
    const bool withSrcValue
)
{
    using GradType = typename detail::LinearUpwindGradType<ValueType>::type;

    const auto exec = src.exec();
    const auto& mesh = src.mesh();

    GaussGreenGrad gradOp(exec, mesh);

    if constexpr (std::is_same_v<ValueType, scalar>)
    {
        // grad(scalar) -> Vec3
        const VolumeField<GradType> gradPhi = gradOp.grad(src);
        applyLinearUpwind<ValueType, GradType>(
            src, flux, faceDeltaOwner, faceDeltaNeighbour, gradPhi, dst, withSrcValue
        );
    }
    else
    {
        // grad(Vec3) -> Tensor
        const VolumeField<GradType> gradPhi = gradOp.gradTensor(src);
        applyLinearUpwind<ValueType, GradType>(
            src, flux, faceDeltaOwner, faceDeltaNeighbour, gradPhi, dst, withSrcValue
        );
    }
}

template<typename ValueType>
void computeLinearUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    SurfaceField<ValueType>& dst
)
{
    computeLinearUpwind<ValueType>(src, flux, faceDeltaOwner, faceDeltaNeighbour, dst, true);
}

template<typename ValueType>
void computeLinearUpwindCorrection(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    SurfaceField<ValueType>& dst
)
{
    computeLinearUpwind<ValueType>(src, flux, faceDeltaOwner, faceDeltaNeighbour, dst, false);
}

#define NF_DECLARE_COMPUTE_IMP_LINUPW_INT(TYPENAME)                                                                                                          \
    template void computeLinearUpwindInterpolation<                                                                                                          \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<Vec3>&, const SurfaceField<Vec3>&, SurfaceField<TYPENAME>&); \
    template void computeLinearUpwindCorrection<                                                                                                             \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<Vec3>&, const SurfaceField<Vec3>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LINUPW_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LINUPW_INT(Vec3);

} // namespace NeoN::finiteVolume::cellCentred
