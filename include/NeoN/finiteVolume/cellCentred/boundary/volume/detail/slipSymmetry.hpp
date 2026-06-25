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
//   vector => tangential projection of the boundary value (the normal component is removed), plus a
//             normal-damping surface-normal gradient that drives the normal component of the cell
//             value towards zero.
//
// The normal damping can be realised two ways, selected by NormalDamping:
//   - Deferred: written into refGrad as -deltaCoeffs*(v·n)*n, so it enters the per-component RHS
//     through the existing fixed-gradient assembly. Keeps the shared scalar matrix + multi-RHS
//     solve, at the cost of lagging the normal coupling by one outer iteration.
//   - Implicit: refGrad is left zero here; the BoundaryAttributes::transformImplicit flag signals
//     the Laplacian assembly to add a per-component diagonal correction (γ|S|·deltaCoeffs·|n_c|)
//     instead, which the solver applies column-by-column. Fully implicit, single matrix, but the
//     solve runs segregated rather than multi-RHS.
//
// NOTE: a future Tensor specialization (e.g. for a transported Reynolds-stress field) must use the
// full reflective transform (T + H·T·H)/2 with H = I - 2 n⊗n, NOT a per-component projection.
namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
{

enum class NormalDamping
{
    Deferred, ///< normal damping via refGrad -> per-component RHS (multi-RHS friendly)
    Implicit  ///< normal damping via per-component diagonal correction (segregated solve)
};

/** @brief Read the opt-in "implicit" flag from a slip/symmetry patch dictionary.
 *
 * Defaults to false (Deferred). A boolean read from a dictionary file arrives as a word/string
 * rather than a bool, so accept bool, int, and the common truthy spellings — mirroring the
 * tolerant parsing used for the solver's l1ScaledResidual flag.
 */
inline bool readTransformImplicit(const Dictionary& dict)
{
    const std::string key = "implicit";
    if (!dict.contains(key)) return true; // [TEST] default slip/symmetry to implicit normal damping
    if (dict.isType<bool>(key)) return dict.get<bool>(key);
    if (dict.isType<int>(key)) return dict.get<int>(key) != 0;
    if (dict.isType<std::string>(key))
    {
        const std::string v = dict.get<std::string>(key);
        return (v == "true" || v == "yes" || v == "on" || v == "1");
    }
    return false;
}

/** @brief Map the implicit flag onto the NormalDamping mode. */
inline NormalDamping normalDampingMode(bool implicit)
{
    return implicit ? NormalDamping::Implicit : NormalDamping::Deferred;
}

// Primary declaration
template<typename ValueType>
void setSlipSymmetryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NormalDamping mode
);

// --- Scalar specialization: zero-gradient (mode is irrelevant; no normal component) ---
template<>
inline void setSlipSymmetryValue<NeoN::scalar>(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    [[maybe_unused]] NormalDamping mode
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

// --- Vec3 specialization: tangential projection + normal damping ---
template<>
inline void setSlipSymmetryValue<NeoN::Vec3>(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NormalDamping mode
)
{
    const auto internalV = domainVector.internalVector().view();
    const bool deferred = (mode == NormalDamping::Deferred);

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

            // Keep the diagonal contribution component-isotropic (zero here) so the shared scalar
            // matrix and its multi-RHS solve are preserved.
            valueFractionV[i] = 0.0;

            // Deferred mode: normal damping as the surface-normal gradient -deltaCoeffs*(v·n)*n
            // (purely normal, so the tangential components stay zero-gradient). Implicit mode:
            // leave refGrad zero — the damping is applied as a per-component diagonal correction
            // during assembly/solve.
            refGradV[i] = deferred ? n * (-deltaCoeffsV[i] * un) : NeoN::zero<NeoN::Vec3>();
        },
        "setSlipSymmetryValue(Vec3)"
    );
}

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
