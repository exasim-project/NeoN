// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

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
    auto dstS = dst.internalField().view();
    const auto [srcS, weightS, ownerS, neighS, boundS] = spans(
        src.internalField(),
        weights.internalField(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        src.boundaryField().value()
    );
    size_t nInternalFaces = dst.mesh().nInternalFaces();

    NeoN::parallelFor(
        exec,
        {0, dstS.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            if (facei < nInternalFaces)
            {
                size_t own = static_cast<size_t>(ownerS[facei]);
                size_t nei = static_cast<size_t>(neighS[facei]);
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
