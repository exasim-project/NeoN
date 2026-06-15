// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linearUpwind.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief applies the upwind value plus the gradient correction for every face.
**
** @tparam ValueType field value type (scalar or Vec3)
** @tparam GradType cell gradient type (Vec3 for scalar fields, Tensor for Vec3 fields)
**
** For both types the correction is `grad & d`: with NeoN's row-major tensor convention
** (gradTensor(i,j) = d U_i / d x_j) the matrix-vector product `Tensor & Vec3` reproduces
** OpenFOAM's `(Cf - C) & gradVf`; for scalars it degenerates to the Vec3 inner product.
*/
template<typename ValueType, typename GradType>
void applyLinearUpwindCorrection(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const VolumeField<GradType>& gradPhi,
    SurfaceField<ValueType>& dst
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
            if (fluxS[facei] >= 0)
            {
                const auto own = ownerS[facei];
                dstS[facei] = srcS[own] + (gradS[own] & dOwnS[facei]);
            }
            else
            {
                const auto nei = neighS[facei];
                dstS[facei] = srcS[nei] + (gradS[nei] & dNeiS[facei]);
            }
        },
        "computeLinearUpwindInterpolationInternal"
    );

    // Physical (non-coupled) boundary faces take the patch value with no correction, matching
    // OpenFOAM's linearUpwind which only corrects coupled patches.
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { dstB[bfi] = boundS[bfi]; },
        "computeLinearUpwindInterpolationBoundary"
    );

    // Processor (coupled) boundary faces: fall back to upwind without the gradient correction.
    // TODO: apply the neighbour-cell gradient correction across rank boundaries (see OpenFOAM).
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces > 0)
    {
        const auto bfOwners = mesh.boundaryMesh().faceOwners().view();
        const auto bFluxV = flux.boundaryData().value().view();
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bcfacei = nBoundaryFaces + procFacei;
                const auto own = bfOwners[bcfacei];
                dstB[bcfacei] = bFluxV[bcfacei] >= 0 ? srcS[own] : boundS[bcfacei];
            },
            "computeLinearUpwindInterpolationProcBoundary"
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
    using GradType = typename detail::LinearUpwindGradType<ValueType>::type;

    const auto exec = src.exec();
    const auto& mesh = src.mesh();
    GaussGreenGrad gradOp(exec, mesh);

    if constexpr (std::is_same_v<ValueType, scalar>)
    {
        // grad(scalar) -> Vec3
        const VolumeField<GradType> gradPhi = gradOp.grad(src);
        applyLinearUpwindCorrection(src, flux, faceDeltaOwner, faceDeltaNeighbour, gradPhi, dst);
    }
    else
    {
        // grad(Vec3) -> Tensor
        const VolumeField<GradType> gradPhi = gradOp.gradTensor(src);
        applyLinearUpwindCorrection(src, flux, faceDeltaOwner, faceDeltaNeighbour, gradPhi, dst);
    }
}

#define NF_DECLARE_COMPUTE_IMP_LINUPW_INT(TYPENAME)                                                \
    template void computeLinearUpwindInterpolation<                                                \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<Vec3>&, const SurfaceField<Vec3>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LINUPW_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LINUPW_INT(Vec3);

} // namespace NeoN::finiteVolume::cellCentred
