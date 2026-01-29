// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::turbulenceModels
{

using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;

class SpalartAllmarasDDES
{
public:

    struct Coefficients
    {
        scalar sigmaNut = 0.66666;
        scalar kappa = 0.41;
        scalar Cb1 = 0.1355;
        scalar Cb2 = 0.622;
        scalar Cw2 = 0.3;
        scalar Cw3 = 2.0;
        scalar Cv1 = 7.1;
        scalar Ct3 = 1.2;
        scalar Ct4 = 0.5;
        scalar Cs = 0.3;
        scalar Cdes = 0.65;
        scalar fdCoef = 8.0;
        scalar fwStar = 0.424;
    };

    SpalartAllmarasDDES(const Executor& exec, const UnstructuredMesh& mesh);

    const Coefficients& coeffs() const;

    scalar cw1() const;

    void omega(
        VolScalarField& omegaField,
        const VolVectorField& gradUx,
        const VolVectorField& gradUy,
        const VolVectorField& gradUz
    ) const;

    void magGradU(
        VolScalarField& magGradUField,
        const VolVectorField& gradUx,
        const VolVectorField& gradUy,
        const VolVectorField& gradUz
    ) const;

    void correctNut(
        VolScalarField& nutField, const VolScalarField& nuTilde, const VolScalarField& nu
    ) const;
    /*
        void computeProdSpDDES(
            VolScalarField& productionField,
            VolScalarField& spCoeffField,
            const VolScalarField& nuTildeField,
            const VolScalarField& nuField,
            const VolScalarField& omegaField,
            const VolScalarField& wallDistanceField,
            const VolScalarField& magGradUField,
            const VolScalarField& deltaField,
            const VolScalarField& gradNuTildeMagSqrField
        ) const;
    */
    void computeProdSpDDES(
        VolScalarField& productionField,
        VolScalarField& spCoeffField,
        const VolScalarField& nuTildeField,
        const VolScalarField& nuField,
        const VolVectorField& gradUx,
        const VolVectorField& gradUy,
        const VolVectorField& gradUz,
        const VolScalarField& wallDistanceField,
        const VolScalarField& deltaField,
        const VolScalarField& gradNuTildeMagSqrField
    ) const;

private:

    Executor exec_;
    const UnstructuredMesh& mesh_;
    Coefficients coeffs_;
    scalar cw1_;
};

} // namespace NeoN::turbulenceModels
