// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLinearInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalVector().view();
    const auto [srcS, weightS, ownerS, neighS, boundS] = views(
        src.internalVector(),
        weights.internalVector(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryData().value()
    );
    auto nInternalFaces = dst.mesh().nInternalFaces();

    NeoN::parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                auto own = ownerS[facei];
                auto nei = neighS[facei];
                dstS[facei] = weightS[facei] * srcS[own] + (1 - weightS[facei]) * srcS[nei];
            }
            else
            {
                dstS[facei] = weightS[facei] * boundS[facei - nInternalFaces];
            }
        }
    );
}

#define NF_DECLARE_COMPUTE_IMP_LIN_INT(TYPENAME)                                                   \
    template void computeLinearInterpolation<                                                      \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LIN_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LIN_INT(Vec3);

// template class Linear<scalar>;
// template class Linear<Vec3>;

} // namespace NeoN
