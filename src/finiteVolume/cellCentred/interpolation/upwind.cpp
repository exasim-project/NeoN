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
    const auto [srcS, weightS, ownerS, neighS, boundS, fluxS] = views(
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
        NEON_LAMBDA(const localIdx facei) {
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


template<typename ValueType>
void computeUpwindInterpolationWeights(
    const SurfaceField<scalar>& flux,
    const VolumeField<ValueType>& src,
    SurfaceField<scalar>& weights
)
{
    const auto exec = src.exec();
    const auto [weightS, weightB, ownerS, neighS, fluxS] = views(
        weights.internalVector(),
        weights.boundaryData().value(),
        src.mesh().faceOwner(),
        src.mesh().faceNeighbour(),
        flux.internalVector()
    );

    NF_ASSERT(flux.internalVector().size() == weights.size(), "different size");

    auto nInternalFaces = src.mesh().nInternalFaces();
    auto nBoundaryFaces = src.mesh().nBoundaryFaces();
    auto totalFaces = flux.size();
    parallelFor(
        exec,
        {0, weights.size()},
        NEON_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                weightS[facei] = fluxS[facei] >= 0 ? 1 : 0;
            }
            auto bcfacei = facei - nInternalFaces;
            if (facei >= nInternalFaces && facei < nInternalFaces + nBoundaryFaces)
            {
                weightB[bcfacei] = 1.0;
                weightS[facei] = 1.0;
            }
            if (facei >= nInternalFaces + nBoundaryFaces && facei < totalFaces)
            {
                auto weight = fluxS[facei] >= 0 ? 1 : 0;
                weightS[facei] = weight;
                weightB[bcfacei] = weight;
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

#define NF_DECLARE_COMPUTE_IMP_UPW_INT_W(TYPENAME)                                                 \
    template void computeUpwindInterpolationWeights<                                               \
        TYPENAME>(const SurfaceField<scalar>&, const VolumeField<TYPENAME>&, SurfaceField<scalar>&)

NF_DECLARE_COMPUTE_IMP_UPW_INT_W(scalar);
NF_DECLARE_COMPUTE_IMP_UPW_INT_W(Vec3);

} // namespace NeoN
