// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/turbulenceModels/DES/SpalartAllmarasBase.hpp"

namespace NeoN::turbulenceModels::DES
{

using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;

class SpalartAllmarasDDES
{
public:

    struct Coefficients
    {
        scalar Cdes = 0.65;
        scalar kappa = 0.41;
        scalar fdCoef = 8.0;
    };

    explicit SpalartAllmarasDDES(const Executor& exec);

    const Coefficients& coeffs() const;

    void dTilde(
        VolScalarField& dTildeField,
        VolScalarField& invSqrdTildeField,
        const VolScalarField& wallDistance,
        const VolScalarField& nuTilde,
        const VolScalarField& nu,
        const VolScalarField& omega,
        const VolScalarField& delta,
        const VolScalarField& chi,
        const VolScalarField& fv1
    ) const;

private:

    Executor exec_;
    Coefficients coeffs_;
};

} // namespace NeoN::turbulenceModels::DES
