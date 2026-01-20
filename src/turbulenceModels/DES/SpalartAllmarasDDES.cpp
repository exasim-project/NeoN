// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/DES/SpalartAllmarasDDES.hpp"

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::turbulenceModels::DES
{

SpalartAllmarasDDES::SpalartAllmarasDDES(const Executor& exec) : exec_(exec) {}

const SpalartAllmarasDDES::Coefficients& SpalartAllmarasDDES::coeffs() const { return coeffs_; }

void SpalartAllmarasDDES::dTilde(
    VolScalarField& dTildeField,
    VolScalarField& invSqrdTildeField,
    const VolScalarField& wallDistance,
    const VolScalarField& nuTilde,
    const VolScalarField& nu,
    const VolScalarField& omega,
    const VolScalarField& delta,
    const VolScalarField& chi,
    const VolScalarField& fv1
) const
{
    const auto& wallVector = wallDistance.internalVector();
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& nuVector = nu.internalVector();
    const auto& omegaVector = omega.internalVector();
    const auto& deltaVector = delta.internalVector();
    const auto& chiVector = chi.internalVector();
    const auto& fv1Vector = fv1.internalVector();
    auto& dTildeVector = dTildeField.internalVector();
    auto& invSqrdTildeVector = invSqrdTildeField.internalVector();

    NF_DEBUG_ASSERT(dTildeVector.size() == wallVector.size(), "dTilde field size mismatch.");
    NF_DEBUG_ASSERT(nuTildeVector.size() == wallVector.size(), "nuTilde field size mismatch.");
    NF_DEBUG_ASSERT(nuVector.size() == wallVector.size(), "nu field size mismatch.");
    NF_DEBUG_ASSERT(omegaVector.size() == wallVector.size(), "omega size mismatch.");
    NF_DEBUG_ASSERT(deltaVector.size() == wallVector.size(), "delta field size mismatch.");

    const auto
        [wallView,
         nuTildeView,
         nuView,
         omegaView,
         deltaView,
         chiView,
         fv1View,
         dTildeView,
         invSqrdTildeView] =
            views(
                wallVector,
                nuTildeVector,
                nuVector,
                omegaVector,
                deltaVector,
                chiVector,
                fv1Vector,
                dTildeVector,
                invSqrdTildeVector
            );

    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar fdCoef = coeffs_.fdCoef;
    const scalar Cdes = coeffs_.Cdes;

    parallelFor(
        exec_,
        {0, dTildeVector.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar d = Kokkos::max(wallView[celli], scalar(1e-18));
            const scalar denom = kappa2 * d * d * (Kokkos::max(omegaView[celli], scalar(1e-18)));
            const scalar rD = Kokkos::min((nuTildeView[celli] + nuView[celli]) / denom, scalar(10));
            const scalar fD = 1.0 - std::tanh(std::pow(fdCoef * rD, 3.0));
            const scalar fv2 = 1.0 - chiView[celli] / (scalar(1) + chiView[celli] * fv1View[celli]);
            const scalar psi = sqrt(Kokkos::min(
                scalar(100),
                (1
                 - scalar(0.1355)
                       / (scalar(3.2390678168) * scalar(0.41) * scalar(0.41) * scalar(0.424)) * fv2)
                    / Kokkos::max(scalar(1e-30), fv1View[celli])
            ));
            const scalar lesDelta = psi * Cdes * deltaView[celli];
            const scalar dDes = Kokkos::max(d - fD * Kokkos::max(0.0, d - lesDelta), scalar(1e-18));
            dTildeView[celli] = dDes;
            invSqrdTildeView[celli] = 1 / (dDes * dDes);
        },
        "SpalartAllmarasDDES::dTilde"
    );
}

} // namespace NeoN::turbulenceModels::DES
