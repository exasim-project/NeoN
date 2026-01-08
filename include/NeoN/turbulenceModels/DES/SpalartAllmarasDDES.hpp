// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"

namespace NeoN::turbulenceModels::DES
{

class SpalartAllmarasDDES
{
public:

    struct Coefficients
    {
        scalar Cdes = 0.65;
        scalar kappa = 0.41;
        scalar fdCoef = 8.0;
    };

    explicit SpalartAllmarasDDES(const Executor& exec, Coefficients coeffs);

    const Coefficients& coeffs() const;

    Vector<scalar> dTilde(
        const Vector<scalar>& wallDistance,
        const Vector<scalar>& nuTilde,
        const Vector<scalar>& nu,
        const Vector<scalar>& strainRate,
        const Vector<scalar>& delta
    ) const;

    void dTilde(
        Vector<scalar>& dTildeField,
        const Vector<scalar>& wallDistance,
        const Vector<scalar>& nuTilde,
        const Vector<scalar>& nu,
        const Vector<scalar>& strainRate,
        const Vector<scalar>& delta
    ) const;

private:

    Executor exec_;
    Coefficients coeffs_;
};

} // namespace NeoN::turbulenceModels::DES
