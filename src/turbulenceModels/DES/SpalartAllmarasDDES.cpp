// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/DES/SpalartAllmarasDDES.hpp"

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/view.hpp"

#include <algorithm>
#include <cmath>

namespace NeoN::turbulenceModels::DES
{

SpalartAllmarasDDES::SpalartAllmarasDDES(const Executor& exec, Coefficients coeffs)
    : exec_(exec), coeffs_(coeffs)
{}

const SpalartAllmarasDDES::Coefficients& SpalartAllmarasDDES::coeffs() const { return coeffs_; }

Vector<scalar> SpalartAllmarasDDES::dTilde(
    const Vector<scalar>& wallDistance,
    const Vector<scalar>& nuTilde,
    const Vector<scalar>& nu,
    const Vector<scalar>& strainRate,
    const Vector<scalar>& delta
) const
{
    Vector<scalar> result(exec_, wallDistance.size());
    dTilde(result, wallDistance, nuTilde, nu, strainRate, delta);
    return result;
}

void SpalartAllmarasDDES::dTilde(
    Vector<scalar>& dTildeField,
    const Vector<scalar>& wallDistance,
    const Vector<scalar>& nuTilde,
    const Vector<scalar>& nu,
    const Vector<scalar>& strainRate,
    const Vector<scalar>& delta
) const
{
    NF_DEBUG_ASSERT(dTildeField.size() == wallDistance.size(), "dTilde field size mismatch.");
    NF_DEBUG_ASSERT(nuTilde.size() == wallDistance.size(), "nuTilde field size mismatch.");
    NF_DEBUG_ASSERT(nu.size() == wallDistance.size(), "nu field size mismatch.");
    NF_DEBUG_ASSERT(strainRate.size() == wallDistance.size(), "strainRate size mismatch.");
    NF_DEBUG_ASSERT(delta.size() == wallDistance.size(), "delta field size mismatch.");

    const auto [wallView, nuTildeView, nuView, strainView, deltaView, dTildeView] =
        views(wallDistance, nuTilde, nu, strainRate, delta, dTildeField);

    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar fdCoef = coeffs_.fdCoef;
    const scalar Cdes = coeffs_.Cdes;

    parallelFor(
        exec_,
        {0, dTildeField.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar d = wallView[celli];
            const scalar denom = kappa2 * d * d * (strainView[celli] + ROOTVSMALL);
            const scalar rD = (nuTildeView[celli] + nuView[celli]) / denom;
            const scalar fD = 1.0 - std::tanh(std::pow(fdCoef * rD, 3.0));
            const scalar lesDelta = Cdes * deltaView[celli];
            const scalar dDes = d - fD * std::max(0.0, d - lesDelta);
            dTildeView[celli] = dDes;
        },
        "SpalartAllmarasDDES::dTilde"
    );
}

} // namespace NeoN::turbulenceModels::DES
