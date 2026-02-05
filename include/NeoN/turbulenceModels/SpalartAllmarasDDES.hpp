// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/tensorVecField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::turbulenceModels
{

using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
using VolScalarField = NeoN::finiteVolume::cellCentred::VolumeField<scalar>;
using SurfScalarField = NeoN::finiteVolume::cellCentred::SurfaceField<scalar>;

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

    void correctNut(
        VolScalarField& nutField,
        SurfScalarField& nutF,
        SurfScalarField& nuEff,
        const VolScalarField& nuTilde,
        const VolScalarField& nu,
        const SurfScalarField& nuF
    ) const;

    void calcNuTildaDiffusionCoeff(
        VolScalarField& nuTilde,
        const SurfScalarField& nuF,
        SurfScalarField& surfNuTilde,
        SurfScalarField& nuTildeEffF
    ) const;

    void calcMagSqrVec(VolScalarField& magSqr, const VolVectorField& in) const;

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
