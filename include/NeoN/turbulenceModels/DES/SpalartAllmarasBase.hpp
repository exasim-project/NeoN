// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::turbulenceModels::DES
{

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

    SpalartAllmarasBase(const Executor& exec, const UnstructuredMesh& mesh, Coefficients coeffs);

    const Coefficients& coeffs() const;

    scalar cw1() const;

    Vector<scalar> wallDistance() const;

    Vector<scalar> strainRate(
        const Vector<Vec3>& gradUx, const Vector<Vec3>& gradUy, const Vector<Vec3>& gradUz
    ) const;

    void strainRate(
        Vector<scalar>& strainRateField,
        const Vector<Vec3>& gradUx,
        const Vector<Vec3>& gradUy,
        const Vector<Vec3>& gradUz
    ) const;

    Vector<scalar> chi(const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const;

    void
    chi(Vector<scalar>& chiField, const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const;

    Vector<scalar> fv1(const Vector<scalar>& chiField) const;

    void fv1(Vector<scalar>& fv1Field, const Vector<scalar>& chiField) const;

    Vector<scalar> fv2(const Vector<scalar>& chiField, const Vector<scalar>& fv1Field) const;

    void
    fv2(Vector<scalar>& fv2Field, const Vector<scalar>& chiField, const Vector<scalar>& fv1Field
    ) const;

    Vector<scalar> ft2(const Vector<scalar>& chiField) const;

    void ft2(Vector<scalar>& ft2Field, const Vector<scalar>& chiField) const;

    Vector<scalar> nut(const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const;

    void
    nut(Vector<scalar>& nutField, const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const;

private:

    Executor exec_;
    const UnstructuredMesh& mesh_;
    Coefficients coeffs_;
    scalar cw1_;
};

} // namespace NeoN::turbulenceModels::DES
