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

SpalartAllmarasBase::SpalartAllmarasBase(const Executor& exec, const UnstructuredMesh& mesh)
    : exec_(exec), mesh_(mesh),
      cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{}

const SpalartAllmarasBase::Coefficients& SpalartAllmarasBase::coeffs() const { return coeffs_; }

scalar SpalartAllmarasBase::cw1() const { return cw1_; }

void SpalartAllmarasBase::wallDistance(VolScalarField& wallDistanceField) const
{
    const auto cellCentresView = mesh_.cellCentres().view();
    const auto boundaryFaceCentresView = mesh_.boundaryMesh().cf().view();
    const auto nCells = mesh_.nCells();
    const auto nBoundaryFaces = mesh_.nBoundaryFaces();
    auto wallDistanceView = wallDistanceField.internalVector().view();

    parallelFor(
        exec_,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const auto cellCentre = cellCentresView[celli];
            scalar minDistance = std::numeric_limits<scalar>::max();
            for (localIdx facei = 0; facei < nBoundaryFaces; ++facei)
            {
                const auto faceCentre = boundaryFaceCentresView[facei];
                const scalar dx = cellCentre[0] - faceCentre[0];
                const scalar dy = cellCentre[1] - faceCentre[1];
                const scalar dz = cellCentre[2] - faceCentre[2];
                const scalar distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                minDistance = std::min(minDistance, distance);
            }
            wallDistanceView[celli] = minDistance;
        },
        "SpalartAllmarasBase::wallDistance"
    );
}

void SpalartAllmarasBase::strainRate(
    VolScalarField& strainRateField,
    const VolVectorField& gradUx,
    const VolVectorField& gradUy,
    const VolVectorField& gradUz
) const
{
    const auto& gradUxVector = gradUx.internalVector();
    const auto& gradUyVector = gradUy.internalVector();
    const auto& gradUzVector = gradUz.internalVector();
    auto& strainVector = strainRateField.internalVector();

    NF_DEBUG_ASSERT(strainVector.size() == gradUxVector.size(), "strainRate size mismatch.");
    NF_DEBUG_ASSERT(gradUyVector.size() == gradUxVector.size(), "gradUy size mismatch.");
    NF_DEBUG_ASSERT(gradUzVector.size() == gradUxVector.size(), "gradUz size mismatch.");

    const auto [gradUxView, gradUyView, gradUzView, strainView] =
        views(gradUxVector, gradUyVector, gradUzVector, strainVector);

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

void SpalartAllmarasBase::chi(
    VolScalarField& chiField, const VolScalarField& nuTilde, const VolScalarField& nu
) const
{
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& nuVector = nu.internalVector();
    auto& chiVector = chiField.internalVector();

    NF_DEBUG_ASSERT(chiVector.size() == nuTildeVector.size(), "chi size mismatch.");
    NF_DEBUG_ASSERT(nuVector.size() == nuTildeVector.size(), "nu size mismatch.");

    const auto [nuTildeView, nuView, chiView] = views(nuTildeVector, nuVector, chiVector);

    parallelFor(
        exec_,
        {0, chiField.size()},
        NEON_LAMBDA(const localIdx celli) {
            chiView[celli] = nuTildeView[celli] / (nuView[celli] + ROOTVSMALL);
        },
        "SpalartAllmarasBase::chi"
    );
}

void SpalartAllmarasBase::fv1(VolScalarField& fv1Field, const VolScalarField& chiField) const
{
    const auto& chiVector = chiField.internalVector();
    auto& fv1Vector = fv1Field.internalVector();

    NF_DEBUG_ASSERT(fv1Vector.size() == chiVector.size(), "fv1 size mismatch.");

    const auto chiView = chiVector.view();
    const auto fv1View = fv1Vector.view();
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

void SpalartAllmarasBase::fv2(
    VolScalarField& fv2Field, const VolScalarField& chiField, const VolScalarField& fv1Field
) const
{
    const auto& chiVector = chiField.internalVector();
    const auto& fv1Vector = fv1Field.internalVector();
    auto& fv2Vector = fv2Field.internalVector();

    NF_DEBUG_ASSERT(fv2Vector.size() == chiVector.size(), "fv2 size mismatch.");
    NF_DEBUG_ASSERT(fv1Vector.size() == chiVector.size(), "fv1 size mismatch.");

    const auto [chiView, fv1View, fv2View] = views(chiVector, fv1Vector, fv2Vector);

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

void SpalartAllmarasBase::ft2(VolScalarField& ft2Field, const VolScalarField& chiField) const
{
    const auto& chiVector = chiField.internalVector();
    auto& ft2Vector = ft2Field.internalVector();

    NF_DEBUG_ASSERT(ft2Vector.size() == chiVector.size(), "ft2 size mismatch.");

    const auto chiView = chiVector.view();
    const auto ft2View = ft2Vector.view();
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

void SpalartAllmarasBase::stilda(
    VolScalarField& stildaField,
    const VolScalarField& strainRate,
    const VolScalarField& nuTilde,
    const VolScalarField& dTilde,
    const VolScalarField& fv2Field
) const
{
    const auto& strainVector = strainRate.internalVector();
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& dTildeVector = dTilde.internalVector();
    const auto& fv2Vector = fv2Field.internalVector();
    auto& stildaVector = stildaField.internalVector();

    NF_DEBUG_ASSERT(stildaVector.size() == strainVector.size(), "stilda size mismatch.");
    NF_DEBUG_ASSERT(nuTildeVector.size() == strainVector.size(), "nuTilde size mismatch.");
    NF_DEBUG_ASSERT(dTildeVector.size() == strainVector.size(), "dTilde size mismatch.");
    NF_DEBUG_ASSERT(fv2Vector.size() == strainVector.size(), "fv2 size mismatch.");

    const auto [strainView, nuTildeView, dTildeView, fv2View, stildaView] =
        views(strainVector, nuTildeVector, dTildeVector, fv2Vector, stildaVector);
    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;

    parallelFor(
        exec_,
        {0, stildaField.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar d = dTildeView[celli];
            const scalar denom = kappa2 * d * d + ROOTVSMALL;
            stildaView[celli] = strainView[celli] + fv2View[celli] * nuTildeView[celli] / denom;
        },
        "SpalartAllmarasBase::stilda"
    );
}

void SpalartAllmarasBase::fw(
    VolScalarField& fwField,
    const VolScalarField& stildaField,
    const VolScalarField& dTilde,
    const VolScalarField& nuTilde
) const
{
    const auto& stildaVector = stildaField.internalVector();
    const auto& dTildeVector = dTilde.internalVector();
    const auto& nuTildeVector = nuTilde.internalVector();
    auto& fwVector = fwField.internalVector();

    NF_DEBUG_ASSERT(fwVector.size() == stildaVector.size(), "fw size mismatch.");
    NF_DEBUG_ASSERT(dTildeVector.size() == stildaVector.size(), "dTilde size mismatch.");
    NF_DEBUG_ASSERT(nuTildeVector.size() == stildaVector.size(), "nuTilde size mismatch.");

    const auto [stildaView, dTildeView, nuTildeView, fwView] =
        views(stildaVector, dTildeVector, nuTildeVector, fwVector);
    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar cw2 = coeffs_.Cw2;
    const scalar cw3 = coeffs_.Cw3;
    const scalar cw3Pow6 = std::pow(cw3, 6.0);

    parallelFor(
        exec_,
        {0, fwField.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar d = dTildeView[celli];
            const scalar denom = stildaView[celli] * kappa2 * d * d + ROOTVSMALL;
            const scalar r = std::min(static_cast<scalar>(10.0), nuTildeView[celli] / denom);
            const scalar g = r + cw2 * (std::pow(r, 6.0) - r);
            const scalar gPow6 = std::pow(g, 6.0);
            fwView[celli] = g * std::pow((1.0 + cw3Pow6) / (gPow6 + cw3Pow6), 1.0 / 6.0);
        },
        "SpalartAllmarasBase::fw"
    );
}

void SpalartAllmarasBase::dNuTildeEff(
    VolScalarField& dNuTildeEffField, const VolScalarField& nuTilde, const VolScalarField& nu
) const
{
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& nuVector = nu.internalVector();
    auto& effVector = dNuTildeEffField.internalVector();

    NF_DEBUG_ASSERT(effVector.size() == nuTildeVector.size(), "dNuTildeEff size mismatch.");
    NF_DEBUG_ASSERT(nuVector.size() == nuTildeVector.size(), "nu size mismatch.");

    const auto [nuTildeView, nuView, effView] = views(nuTildeVector, nuVector, effVector);
    const scalar sigmaNut = coeffs_.sigmaNut;

    parallelFor(
        exec_,
        {0, dNuTildeEffField.size()},
        NEON_LAMBDA(const localIdx celli) {
            effView[celli] = (nuTildeView[celli] + nuView[celli]) / sigmaNut;
        },
        "SpalartAllmarasBase::dNuTildeEff"
    );
}

void SpalartAllmarasBase::nut(
    VolScalarField& nutField, const VolScalarField& nuTilde, const VolScalarField& nu
) const
{
    const auto& nuTildeVector = nuTilde.internalVector();
    const auto& nuVector = nu.internalVector();
    auto& nutVector = nutField.internalVector();

    NF_DEBUG_ASSERT(nutVector.size() == nuTildeVector.size(), "nut size mismatch.");
    NF_DEBUG_ASSERT(nuVector.size() == nuTildeVector.size(), "nu size mismatch.");

    Vector<scalar> chiField(exec_, nuTildeVector.size());
    Vector<scalar> fv1Field(exec_, nuTildeVector.size());

    const auto [nuTildeView, nuView, chiView] = views(nuTildeVector, nuVector, chiField);
    parallelFor(
        exec_,
        {0, chiField.size()},
        NEON_LAMBDA(const localIdx celli) {
            chiView[celli] = nuTildeView[celli] / (nuView[celli] + ROOTVSMALL);
        },
        "SpalartAllmarasBase::nutChi"
    );

    const scalar cv1 = coeffs_.Cv1;
    const scalar cv1Cubed = cv1 * cv1 * cv1;
    const auto chiViewForFv1 = chiField.view();
    const auto fv1View = fv1Field.view();
    parallelFor(
        exec_,
        {0, fv1Field.size()},
        NEON_LAMBDA(const localIdx celli) {
            const scalar chiVal = chiViewForFv1[celli];
            const scalar chiCubed = chiVal * chiVal * chiVal;
            fv1View[celli] = chiCubed / (chiCubed + cv1Cubed);
        },
        "SpalartAllmarasBase::nutFv1"
    );

    const auto [nuTildeViewForNut, fv1ViewForNut, nutView] =
        views(nuTildeVector, fv1Field, nutVector);

    parallelFor(
        exec_,
        {0, nutField.size()},
        NEON_LAMBDA(const localIdx celli) { nutView[celli] = nuTildeView[celli] * fv1View[celli]; },
        "SpalartAllmarasBase::nut"
    );
}

} // namespace NeoN::turbulenceModels::DES
