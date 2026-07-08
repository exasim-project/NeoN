// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void surfaceIntegrate(
    const Executor& exec,
    localIdx nInternalFaces,
    View<const int> neighbors,
    View<const int> owners,
    View<const int> faceOwners,
    View<const ValueType> flux,
    View<const ValueType> bFlux,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    auto nCells = v.size();
    const auto nBoundaryFaces = faceOwners.size();
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            Kokkos::atomic_add(&res[owners[i]], flux[i]);
            Kokkos::atomic_sub(&res[neighbors[i]], flux[i]);
        },
        "surfaceIntegrateInternalFaces"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = faceOwners[bfi];
            Kokkos::atomic_add(&res[own], bFlux[bfi]);
        },
        "surfaceIntegrateBoundaryFaces"
    );

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) { res[celli] *= operatorScaling[celli] / v[celli]; },
        "surfaceIntegrateInternalCells"
    );
}

#define NF_DECLARE_COMPUTE_IMP_INT(TYPENAME)                                                       \
    template void surfaceIntegrate<TYPENAME>(                                                      \
        const Executor&,                                                                           \
        localIdx,                                                                                  \
        View<const int>,                                                                           \
        View<const int>,                                                                           \
        View<const int>,                                                                           \
        View<const TYPENAME>,                                                                      \
        View<const TYPENAME>,                                                                      \
        View<const scalar>,                                                                        \
        View<TYPENAME>,                                                                            \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_IMP_INT(scalar);
NF_DECLARE_COMPUTE_IMP_INT(Vec3);

// instantiate the template class
template class SurfaceIntegrate<scalar>;
template class SurfaceIntegrate<Vec3>;

};
