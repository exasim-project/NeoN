// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

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
    const auto [srcS, weightS, ownerS, neighS, boundS, fluxS] = spans(
        src.internalVector(),
        weights.internalVector(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryData().value(),
        flux.internalVector()
    );
    auto nInternalFaces = dst.mesh().nInternalFaces();

    parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                if (fluxS[facei] >= 0)
                {
                    auto own = ownerS[facei];
                    dstS[facei] = srcS[own];
                }
                else
                {
                    auto nei = neighS[facei];
                    dstS[facei] = srcS[nei];
                }
            }
            else
            {
                dstS[facei] = weightS[facei] * boundS[facei - nInternalFaces];
            }
        },
        "computeUpwindInterpolation"
    );
}

#define NF_DECLARE_COMPUTE_IMP_UPW_INT(TYPENAME)                                                   \
    template void computeUpwindInterpolation<                                                      \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_UPW_INT(scalar);
NF_DECLARE_COMPUTE_IMP_UPW_INT(Vec3);


template<typename ValueType>
void computeUpwindInterpolationFace(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const localIdx faceIdx
)
{
    const auto [dstS, srcS, weightS, ownerS, neighS, boundS, fluxS] = spans(
        dst.internalVector(),
        src.internalVector(),
        weights.internalVector(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryData().value(),
        flux.internalVector()
    );
    auto nInternalFaces = dst.mesh().nInternalFaces();

    if (faceIdx < nInternalFaces)
    {
        if (fluxS[faceIdx] >= 0)
        {
            dstS[faceIdx] = srcS[ownerS[faceIdx]];
        }
        else
        {
            dstS[faceIdx] = srcS[neighS[faceIdx]];
        }
    }
    else
    {
        dstS[faceIdx] = weightS[faceIdx] * boundS[faceIdx - nInternalFaces];
    }
}
#define NF_DECLARE_COMPUTE_IMP_UPW_INT_F(TYPENAME)                                                 \
    template void computeUpwindInterpolationFace<TYPENAME>(                                        \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        SurfaceField<TYPENAME>&,                                                                   \
        const localIdx                                                                             \
    )

NF_DECLARE_COMPUTE_IMP_UPW_INT_F(scalar);
NF_DECLARE_COMPUTE_IMP_UPW_INT_F(Vec3);

} // namespace NeoN
