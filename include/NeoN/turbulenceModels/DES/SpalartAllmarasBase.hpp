// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
// #include "NeoN/core/primitives/scalar.hpp"
// #include "NeoN/core/primitives/vec3.hpp"
// #include "NeoN/core/vector/vector.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::turbulenceModels::DES
{

using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;

class SpalartAllmarasBase
{
public:

    struct Coefficients
    {
        scalar sigmaNut = 2.0 / 3.0;
        scalar kappa = 0.41;
        scalar Cb1 = 0.1355;
        scalar Cb2 = 0.622;
        scalar Cw2 = 0.3;
        scalar Cw3 = 2.0;
        scalar Cv1 = 7.1;
        scalar Ct3 = 1.2;
        scalar Ct4 = 0.5;
        scalar Cs = 0.3;
    };

    SpalartAllmarasBase(const Executor& exec, const UnstructuredMesh& mesh);

    const Coefficients& coeffs() const;

    scalar cw1() const;

    void wallDistance(VolScalarField& wallDistanceField) const;

    void strainRate(
        VolScalarField& strainRateField,
        const VolVectorField& gradUx,
        const VolVectorField& gradUy,
        const VolVectorField& gradUz
    ) const;


    void
    chi(VolScalarField& chiField, const VolScalarField& nuTilde, const VolScalarField& nu) const;

    void fv1(VolScalarField& fv1Field, const VolScalarField& chiField) const;

    void
    fv2(VolScalarField& fv2Field, const VolScalarField& chiField, const VolScalarField& fv1Field
    ) const;

    void ft2(VolScalarField& ft2Field, const VolScalarField& chiField) const;

    void stilda(
        VolScalarField& stildaField,
        const VolScalarField& strainRate,
        const VolScalarField& nuTilde,
        const VolScalarField& dTilde,
        const VolScalarField& fv2Field
    ) const;

    void
    fw(VolScalarField& fwField,
       const VolScalarField& stildaField,
       const VolScalarField& dTilde,
       const VolScalarField& nuTilde) const;

    void dNuTildeEff(
        VolScalarField& dNuTildeEffField, const VolScalarField& nuTilde, const VolScalarField& nu
    ) const;

    void
    nut(VolScalarField& nutField, const VolScalarField& nuTilde, const VolScalarField& nu) const;

private:

    Executor exec_;
    const UnstructuredMesh& mesh_;
    Coefficients coeffs_;
    scalar cw1_;
};

} // namespace NeoN::turbulenceModels::DES
