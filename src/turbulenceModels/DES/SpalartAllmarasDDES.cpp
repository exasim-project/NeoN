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

SpalartAllmarasDDES::SpalartAllmarasDDES(const Executor& exec) : exec_(exec) {}

const SpalartAllmarasDDES::Coefficients& SpalartAllmarasDDES::coeffs() const { return coeffs_; }

void SpalartAllmarasDDES::dTilde(
    VolScalarField& dTildeField,
    const VolScalarField& wallDistance,
    const VolScalarField& nuTilde,
    const VolScalarField& nu,
    const VolScalarField& strainRate,
    const VolScalarField& delta
) const
{
    const auto& wallVector = wallDistance.internalVector();
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& nuVector = nu.internalVector();
    const auto& strainVector = strainRate.internalVector();
    const auto& deltaVector = delta.internalVector();
    auto& dTildeVector = dTildeField.internalVector();

    NF_DEBUG_ASSERT(dTildeVector.size() == wallVector.size(), "dTilde field size mismatch.");
    NF_DEBUG_ASSERT(nuTildeVector.size() == wallVector.size(), "nuTilde field size mismatch.");
    NF_DEBUG_ASSERT(nuVector.size() == wallVector.size(), "nu field size mismatch.");
    NF_DEBUG_ASSERT(strainVector.size() == wallVector.size(), "strainRate size mismatch.");
    NF_DEBUG_ASSERT(deltaVector.size() == wallVector.size(), "delta field size mismatch.");

    const auto [wallView, nuTildeView, nuView, strainView, deltaView, dTildeView] =
        views(wallVector, nuTildeVector, nuVector, strainVector, deltaVector, dTildeVector);

    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar fdCoef = coeffs_.fdCoef;
    const scalar Cdes = coeffs_.Cdes;

    parallelFor(
        exec_,
        {0, dTildeVector.size()},
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

void SpalartAllmarasDDES::correctNut(
    VolScalarField& nutField,
    const SpalartAllmarasBase& base,
    const VolScalarField& nuTilde,
    const VolScalarField& nu
) const
{
    base.nut(nutField, nuTilde, nu);
}

void SpalartAllmarasDDES::correct(
    VolScalarField& dTildeField,
    VolScalarField& nutField,
    const SpalartAllmarasBase& base,
    const VolScalarField& wallDistance,
    const VolScalarField& nuTilde,
    const VolScalarField& nu,
    const VolScalarField& strainRate,
    const VolScalarField& delta
) const
{
    dTilde(dTildeField, wallDistance, nuTilde, nu, strainRate, delta);
    correctNut(nutField, base, nuTilde, nu);
}

} // namespace NeoN::turbulenceModels::DES
