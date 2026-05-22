// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/primitives/symmTensor.hpp"

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
    auto dstB = dst.boundaryData().value().view();
    const auto [srcS, weightS, weightB, ownerS, neighS, boundS] = views(
        src.internalVector(),
        weights.internalVector(),
        weights.boundaryData().value(),
        dst.mesh().faceOwners(),
        dst.mesh().faceNeighbors(),
        src.boundaryData().value()
    );
    auto nInternalFaces = dst.mesh().nInternalFaces();
    auto nBoundaryFaces = dst.mesh().nBoundaryFaces();

    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = ownerS[facei];
            auto nei = neighS[facei];
            dstS[facei] = weightS[facei] * srcS[own] + (1 - weightS[facei]) * srcS[nei];
        },
        "computeLinearInterpolationInternal"
    );

    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) { dstB[bfi] = weightB[bfi] * boundS[bfi]; },
        "computeLinearInterpolationBoundary"
    );
}

#define NF_DECLARE_COMPUTE_IMP_LIN_INT(TYPENAME)                                                   \
    template void computeLinearInterpolation<                                                      \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LIN_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LIN_INT(Vec3);
NF_DECLARE_COMPUTE_IMP_LIN_INT(Tensor);
NF_DECLARE_COMPUTE_IMP_LIN_INT(SymmTensor);

// template class Linear<scalar>;
// template class Linear<Vec3>;

} // namespace NeoN
