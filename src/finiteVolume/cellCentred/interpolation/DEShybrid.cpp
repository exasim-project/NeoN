// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/interpolation/DEShybrid.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void blendSurfaceFields(
    const SurfaceField<scalar>& sigma,
    const SurfaceField<ValueType>& a,
    const SurfaceField<ValueType>& b,
    SurfaceField<ValueType>& out
)
{
    const auto exec = out.exec();

    // Internal faces: out = (1 - sigma)*a + sigma*b. scalar*Vec3 and Vec3+Vec3 degenerate to the
    // scalar inner product for scalar fields.
    auto outS = out.internalVector().view();
    const auto [sigS, aS, bS] =
        views(sigma.internalVector(), a.internalVector(), b.internalVector());
    parallelFor(
        exec,
        {0, outS.size()},
        NEON_LAMBDA(const localIdx i) {
            outS[i] = (scalar(1) - sigS[i]) * aS[i] + sigS[i] * bS[i];
        },
        "blendSurfaceFieldsInternal"
    );

    // Boundary faces (physical + processor) share the same blend.
    auto outB = out.boundaryData().value().view();
    const auto [sigB, aB, bB] =
        views(sigma.boundaryData().value(), a.boundaryData().value(), b.boundaryData().value());
    parallelFor(
        exec,
        {0, outB.size()},
        NEON_LAMBDA(const localIdx i) {
            outB[i] = (scalar(1) - sigB[i]) * aB[i] + sigB[i] * bB[i];
        },
        "blendSurfaceFieldsBoundary"
    );
}

#define NF_DECLARE_BLEND_SURFACE_FIELDS(TYPENAME)                                                  \
    template void blendSurfaceFields<                                                              \
        TYPENAME>(const SurfaceField<scalar>&, const SurfaceField<TYPENAME>&, const SurfaceField<TYPENAME>&, SurfaceField<TYPENAME>&)

NF_DECLARE_BLEND_SURFACE_FIELDS(scalar);
NF_DECLARE_BLEND_SURFACE_FIELDS(Vec3);

} // namespace NeoN::finiteVolume::cellCentred
