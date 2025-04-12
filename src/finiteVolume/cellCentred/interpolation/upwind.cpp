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
    auto dstS = dst.internalField().view();
    const auto [srcS, weightS, ownerS, neighS, boundS, fluxS] = spans(
        src.internalField(),
        weights.internalField(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryField().value(),
        flux.internalField()
    );
    size_t nInternalFaces = dst.mesh().nInternalFaces();

    NeoN::parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            if (facei < nInternalFaces)
            {
                if (fluxS[facei] >= 0)
                {
                    size_t own = static_cast<size_t>(ownerS[facei]);
                    dstS[facei] = srcS[own];
                }
                else
                {
                    size_t nei = static_cast<size_t>(neighS[facei]);
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
NF_DECLARE_COMPUTE_IMP_UPW_INT(Vector);

} // namespace NeoN
