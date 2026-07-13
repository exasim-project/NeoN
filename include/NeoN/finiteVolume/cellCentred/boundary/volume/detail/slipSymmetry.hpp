// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

// Shared implementation behind the `slip` and `symmetry` boundary conditions: they apply the same
// operator and differ only in their registered name and in where they may be applied (slip on a
// wall/regular patch, symmetry on a symmetry-plane patch). The operator is, per face:
//   scalar => zero-gradient
//   vector => tangential projection of the boundary value (the normal component is removed), plus
//             a deferred normal-damping surface-normal gradient -deltaCoeffs*(v·n)*n that drives
//             the normal component of the cell value towards zero via the per-component RHS.
namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
{

// Primary declaration
template<typename ValueType>
void setSlipSymmetryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
);

// --- Scalar specialization: zero-gradient ---
template<>
inline void setSlipSymmetryValue<NeoN::scalar>(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internalV = domainVector.internalVector().view();

    auto [refGradV, valueV, valueFractionV, refValueV, boundaryFaceOwnersV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const auto v = internalV[owner];

            refValueV[i] = v;
            valueV[i] = v;
            valueFractionV[i] = 0.0;
            refGradV[i] = 0.0;
        },
        "setSlipSymmetryValue(scalar)"
    );
}

// --- Vec3 specialization: tangential projection + deferred normal damping ---
template<>
inline void setSlipSymmetryValue<NeoN::Vec3>(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internalV = domainVector.internalVector().view();

    auto
        [refGradV,
         valueV,
         valueFractionV,
         refValueV,
         boundaryFaceOwnersV,
         faceUnitNormalsV,
         deltaCoeffsV] =
            views(
                domainVector.boundaryData().refGrad(),
                domainVector.boundaryData().value(),
                domainVector.boundaryData().valueFraction(),
                domainVector.boundaryData().refValue(),
                mesh.boundaryMesh().faceOwners(),
                mesh.boundaryMesh().faceUnitNormals(),
                mesh.boundaryMesh().deltaCoeffs()
            );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const auto v = internalV[owner];
            const auto n = faceUnitNormalsV[i];

            const auto un = (v & n);      // normal component (scalar)
            const auto vtan = v - n * un; // tangential projection (remove normal component)

            // Explicit boundary value: used by convection / face interpolation (no penetration).
            refValueV[i] = vtan;
            valueV[i] = vtan;

            // valueFraction = 0: keep the diagonal contribution component-isotropic so the shared
            // scalar matrix and multi-RHS solve are preserved.
            valueFractionV[i] = 0.0;

            // Normal damping via the surface-normal gradient -deltaCoeffs*(v·n)*n enters the
            // per-component RHS through the existing fixed-gradient assembly.
            refGradV[i] = n * (-deltaCoeffsV[i] * un);
        },
        "setSlipSymmetryValue(Vec3)"
    );
}

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
