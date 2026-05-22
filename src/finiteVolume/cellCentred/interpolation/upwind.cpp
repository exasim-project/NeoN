// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalVector().view();
    auto dstB = dst.boundaryData().value().view();
    const auto [srcS, weightB, ownerS, neighS, boundS, fluxS] = views(
        src.internalVector(),
        weights.boundaryData().value(),
        dst.mesh().faceOwners(),
        dst.mesh().faceNeighbors(),
        src.boundaryData().value(),
        flux.internalVector()
    );
    auto nInternalFaces = dst.mesh().nInternalFaces();
    auto nBoundaryFaces = dst.mesh().nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            if (fluxS[facei] >= 0)
            {
                dstS[facei] = srcS[ownerS[facei]];
            }
            else
            {
                dstS[facei] = srcS[neighS[facei]];
            }
        },
        "computeUpwindInterpolationInternal"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { dstB[bfi] = weightB[bfi] * boundS[bfi]; },
        "computeUpwindInterpolationBoundary"
    );
}


template<typename ValueType>
void computeUpwindInterpolationWeights(
    const SurfaceField<scalar>& flux,
    const VolumeField<ValueType>& src,
    SurfaceField<scalar>& weights
)
{
    const auto exec = src.exec();
    const auto [weightS, weightB, fluxS] =
        views(weights.internalVector(), weights.boundaryData().value(), flux.internalVector());
    auto nInternalFaces = src.mesh().nInternalFaces();
    auto nBoundaryFaces = src.mesh().nBoundaryFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) { weightS[facei] = fluxS[facei] >= 0 ? 1 : 0; },
        "computeUpwindWeightsInternal"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { weightB[bfi] = 1.0; },
        "computeUpwindWeightsBoundary"
    );
}

#define NF_DECLARE_COMPUTE_IMP_UPW_INT(TYPENAME)                                                   \
    template void computeUpwindInterpolation<                                                      \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_UPW_INT(scalar);
NF_DECLARE_COMPUTE_IMP_UPW_INT(Vec3);

#define NF_DECLARE_COMPUTE_IMP_UPW_INT_W(TYPENAME)                                                 \
    template void computeUpwindInterpolationWeights<                                               \
        TYPENAME>(const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, SurfaceField<scalar>&)

NF_DECLARE_COMPUTE_IMP_UPW_INT_W(scalar);
NF_DECLARE_COMPUTE_IMP_UPW_INT_W(Vec3);

} // namespace NeoN
