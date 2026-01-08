// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/DES/SpalartAllmarasBase.hpp"

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"

// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <vector>

namespace NeoN::turbulenceModels::DES
{

SpalartAllmarasBase::SpalartAllmarasBase(
    const Executor& exec, const UnstructuredMesh& mesh, Coefficients coeffs
)
    : exec_(exec), mesh_(mesh), coeffs_(coeffs),
      cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{}

const SpalartAllmarasBase::Coefficients& SpalartAllmarasBase::coeffs() const { return coeffs_; }

scalar SpalartAllmarasBase::cw1() const { return cw1_; }

Vector<scalar> SpalartAllmarasBase::wallDistance() const
{
    const auto cellCentresHost = mesh_.cellCentres().copyToHost();
    const auto boundaryFaceCentresHost = mesh_.boundaryMesh().cf().copyToHost();

    const auto cellCentresView = cellCentresHost.view();
    const auto boundaryFaceCentresView = boundaryFaceCentresHost.view();
    const auto nCells = mesh_.nCells();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();

    std::vector<scalar> distances(nCells, std::numeric_limits<scalar>::max());

    for (localIdx celli = 0; celli < nCells; ++celli)
    {
        const auto cellCentre = cellCentresView[celli];
        scalar minDistance = distances[celli];
        for (localIdx facei = 0; facei < nBoundaryFaces; ++facei)
        {
            const auto faceCentre = boundaryFaceCentresView[facei];
            const scalar dx = cellCentre[0] - faceCentre[0];
            const scalar dy = cellCentre[1] - faceCentre[1];
            const scalar dz = cellCentre[2] - faceCentre[2];
            const scalar distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            minDistance = std::min(minDistance, distance);
        }
        distances[celli] = minDistance;
    }

    return Vector<scalar>(exec_, distances);
}

Vector<scalar> SpalartAllmarasBase::strainRate(
    const Vector<Vec3>& gradUx, const Vector<Vec3>& gradUy, const Vector<Vec3>& gradUz
) const
{
    Vector<scalar> result(exec_, gradUx.size());
    strainRate(result, gradUx, gradUy, gradUz);
    return result;
}

void SpalartAllmarasBase::strainRate(
    Vector<scalar>& strainRateField,
    const Vector<Vec3>& gradUx,
    const Vector<Vec3>& gradUy,
    const Vector<Vec3>& gradUz
) const
{
    NF_DEBUG_ASSERT(strainRateField.size() == gradUx.size(), "strainRate size mismatch.");
    NF_DEBUG_ASSERT(gradUy.size() == gradUx.size(), "gradUy size mismatch.");
    NF_DEBUG_ASSERT(gradUz.size() == gradUx.size(), "gradUz size mismatch.");

    const auto [gradUxView, gradUyView, gradUzView, strainView] =
        views(gradUx, gradUy, gradUz, strainRateField);

    parallelFor(
        exec_,
        {0, strainRateField.size()},
        NEON_LAMBDA(const localIdx celli) {
            const auto du = gradUxView[celli];
            const auto dv = gradUyView[celli];
            const auto dw = gradUzView[celli];

            const scalar s11 = du[0];
            const scalar s22 = dv[1];
            const scalar s33 = dw[2];
            const scalar s12 = 0.5 * (du[1] + dv[0]);
            const scalar s13 = 0.5 * (du[2] + dw[0]);
            const scalar s23 = 0.5 * (dv[2] + dw[1]);

            const scalar sInner =
                s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
            strainView[celli] = std::sqrt(2.0 * sInner);
        },
        "SpalartAllmarasBase::strainRate"
    );
}

Vector<scalar>
SpalartAllmarasBase::chi(const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const
{
    Vector<scalar> result(exec_, nuTilde.size());
    chi(result, nuTilde, nu);
    return result;
}

void SpalartAllmarasBase::chi(
    Vector<scalar>& chiField, const Vector<scalar>& nuTilde, const Vector<scalar>& nu
) const
{
    NF_DEBUG_ASSERT(chiField.size() == nuTilde.size(), "chi size mismatch.");
    NF_DEBUG_ASSERT(nu.size() == nuTilde.size(), "nu size mismatch.");

    const auto [nuTildeView, nuView, chiView] = views(nuTilde, nu, chiField);

    parallelFor(
        exec_,
        {0, chiField.size()},
        NEON_LAMBDA(const localIdx celli) {
            chiView[celli] = nuTildeView[celli] / (nuView[celli] + ROOTVSMALL);
        },
        "SpalartAllmarasBase::chi"
    );
}

Vector<scalar> SpalartAllmarasBase::fv1(const Vector<scalar>& chiField) const
{
    Vector<scalar> result(exec_, chiField.size());
    fv1(result, chiField);
    return result;
}

void SpalartAllmarasBase::fv1(Vector<scalar>& fv1Field, const Vector<scalar>& chiField) const
{
    NF_DEBUG_ASSERT(fv1Field.size() == chiField.size(), "fv1 size mismatch.");

    const auto chiView = chiField.view();
    const auto fv1View = fv1Field.view();
    const scalar cv1 = coeffs_.Cv1;
    const scalar cv1Cubed = cv1 * cv1 * cv1;

    parallelFor(
        exec_,
        {0, fv1Field.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar chiVal = chiView[celli];
            const scalar chiCubed = chiVal * chiVal * chiVal;
            fv1View[celli] = chiCubed / (chiCubed + cv1Cubed);
        },
        "SpalartAllmarasBase::fv1"
    );
}

Vector<scalar>
SpalartAllmarasBase::fv2(const Vector<scalar>& chiField, const Vector<scalar>& fv1Field) const
{
    Vector<scalar> result(exec_, chiField.size());
    fv2(result, chiField, fv1Field);
    return result;
}

void SpalartAllmarasBase::fv2(
    Vector<scalar>& fv2Field, const Vector<scalar>& chiField, const Vector<scalar>& fv1Field
) const
{
    NF_DEBUG_ASSERT(fv2Field.size() == chiField.size(), "fv2 size mismatch.");
    NF_DEBUG_ASSERT(fv1Field.size() == chiField.size(), "fv1 size mismatch.");

    const auto [chiView, fv1View, fv2View] = views(chiField, fv1Field, fv2Field);

    parallelFor(
        exec_,
        {0, fv2Field.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar chiVal = chiView[celli];
            fv2View[celli] = 1.0 - chiVal / (1.0 + chiVal * fv1View[celli]);
        },
        "SpalartAllmarasBase::fv2"
    );
}

Vector<scalar> SpalartAllmarasBase::ft2(const Vector<scalar>& chiField) const
{
    Vector<scalar> result(exec_, chiField.size());
    ft2(result, chiField);
    return result;
}

void SpalartAllmarasBase::ft2(Vector<scalar>& ft2Field, const Vector<scalar>& chiField) const
{
    NF_DEBUG_ASSERT(ft2Field.size() == chiField.size(), "ft2 size mismatch.");

    const auto chiView = chiField.view();
    const auto ft2View = ft2Field.view();
    const scalar ct3 = coeffs_.Ct3;
    const scalar ct4 = coeffs_.Ct4;

    parallelFor(
        exec_,
        {0, ft2Field.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar chiVal = chiView[celli];
            ft2View[celli] = ct3 * std::exp(-ct4 * chiVal * chiVal);
        },
        "SpalartAllmarasBase::ft2"
    );
}

Vector<scalar>
SpalartAllmarasBase::nut(const Vector<scalar>& nuTilde, const Vector<scalar>& nu) const
{
    Vector<scalar> result(exec_, nuTilde.size());
    nut(result, nuTilde, nu);
    return result;
}

void SpalartAllmarasBase::nut(
    Vector<scalar>& nutField, const Vector<scalar>& nuTilde, const Vector<scalar>& nu
) const
{
    NF_DEBUG_ASSERT(nutField.size() == nuTilde.size(), "nut size mismatch.");
    NF_DEBUG_ASSERT(nu.size() == nuTilde.size(), "nu size mismatch.");

    Vector<scalar> chiField(exec_, nuTilde.size());
    Vector<scalar> fv1Field(exec_, nuTilde.size());
    chi(chiField, nuTilde, nu);
    fv1(fv1Field, chiField);

    const auto [nuTildeView, fv1View, nutView] = views(nuTilde, fv1Field, nutField);

    parallelFor(
        exec_,
        {0, nutField.size()},
        NEON_LAMBDA(const localIdx celli) { nutView[celli] = nuTildeView[celli] * fv1View[celli]; },
        "SpalartAllmarasBase::nut"
    );
}

} // namespace NeoN::turbulenceModels::DES
