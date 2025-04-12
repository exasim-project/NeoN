// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors


#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void surfaceIntegrate(
    const Executor& exec,
    size_t nInternalFaces,
    View<const int> neighbour,
    View<const int> owner,
    View<const int> faceCells,
    View<const ValueType> flux,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    size_t nCells {v.size()};
    const size_t nBoundaryFaces = faceCells.size();
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Kokkos::atomic_add(&res[static_cast<size_t>(owner[i])], flux[i]);
            Kokkos::atomic_sub(&res[static_cast<size_t>(neighbour[i])], flux[i]);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + nBoundaryFaces},
        KOKKOS_LAMBDA(const size_t i) {
            auto own = static_cast<size_t>(faceCells[i - nInternalFaces]);
            Kokkos::atomic_add(&res[own], flux[i]);
        }
    );

    parallelFor(
        exec,
        {0, nCells},
        KOKKOS_LAMBDA(const size_t celli) { res[celli] *= operatorScaling[celli] / v[celli]; }
    );
}

#define NF_DECLARE_COMPUTE_IMP_INT(TYPENAME)                                                       \
    template void surfaceIntegrate<TYPENAME>(                                                      \
        const Executor&,                                                                           \
        size_t,                                                                                    \
        View<const int>,                                                                           \
        View<const int>,                                                                           \
        View<const int>,                                                                           \
        View<const TYPENAME>,                                                                      \
        View<const scalar>,                                                                        \
        View<TYPENAME>,                                                                            \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_IMP_INT(scalar);
NF_DECLARE_COMPUTE_IMP_INT(Vector);

// instantiate the template class
template class SurfaceIntegrate<scalar>;
template class SurfaceIntegrate<Vector>;

};
