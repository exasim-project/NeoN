// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/SpalartAllmarasDDES.hpp"

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::turbulenceModels
{

SpalartAllmarasDDES::SpalartAllmarasDDES(const Executor& exec, const UnstructuredMesh& mesh)
    : exec_(exec), mesh_(mesh),
      cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{}

const SpalartAllmarasDDES::Coefficients& SpalartAllmarasDDES::coeffs() const { return coeffs_; }

scalar SpalartAllmarasDDES::cw1() const { return cw1_; }

void SpalartAllmarasDDES::correctNut(
    VolScalarField& nutField, const VolScalarField& nuTilde, const VolScalarField& nu
) const
{
    // --- Internal data
    const auto& nuTildeI = nuTilde.internalVector();
    const auto& nuI = nu.internalVector();
    auto& nutI = nutField.internalVector();

    NF_DEBUG_ASSERT(nuTildeI.size() == nuI.size(), "nuTilde / nu size mismatch");
    NF_DEBUG_ASSERT(nutI.size() == nuTildeI.size(), "nut size mismatch");

    const scalar cv1 = coeffs_.Cv1;
    const scalar cv1Cubed = cv1 * cv1 * cv1;

    const auto [nuTildeV, nuV, nutV] = views(nuTildeI, nuI, nutI);

    parallelFor(
        exec_,
        {0, nutI.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar chi = nuTildeV[i] / nuV[i];
            const scalar chi3 = chi * chi * chi;
            nutV[i] = nuTildeV[i] * chi3 / (chi3 + cv1Cubed);
        },
        "SA-DDES::correctNut::internal"
    );

    // --- Boundary handling stays explicit and unchanged
    nutField.correctBoundaryConditions();
}

void SpalartAllmarasDDES::correctNut(
    VolScalarField& nutField,
    SurfScalarField& nutF,
    SurfScalarField& nuEffF,
    const VolScalarField& nuTilde,
    const VolScalarField& nu,
    const SurfScalarField& nuF
) const
{
    // --- Internal data
    const auto& nuTildeI = nuTilde.internalVector();
    const auto& nuI = nu.internalVector();
    auto& nutI = nutField.internalVector();

    NF_DEBUG_ASSERT(nuTildeI.size() == nuI.size(), "nuTilde / nu size mismatch");
    NF_DEBUG_ASSERT(nutI.size() == nuTildeI.size(), "nut size mismatch");

    const scalar cv1 = coeffs_.Cv1;
    const scalar cv1Cubed = cv1 * cv1 * cv1;

    const auto [nuTildeV, nuV, nutV] = views(nuTildeI, nuI, nutI);

    // --- Internal nut update
    parallelFor(
        exec_,
        {0, nutI.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar chi = nuTildeV[i] / nuV[i];
            const scalar chi3 = chi * chi * chi;
            nutV[i] = nuTildeV[i] * chi3 / (chi3 + cv1Cubed);
        },
        "SA-DDES::correctNut::internal"
    );

    // --- Boundary conditions MUST be applied before face usage
    nutField.correctBoundaryConditions();

    // --- Interpolate nut to faces (API-correct form)
    fvcc::SurfaceInterpolation<scalar> surfInterp(
        exec_, nutField.mesh(), NeoN::TokenList({std::string("linear")})
    );

    surfInterp.interpolate(nutField, nutF);

    // --- Build nuEffF = nuF + nutF (face algebra)
    {
        const auto& nuFI = nuF.internalVector();
        const auto& nutFI = nutF.internalVector();
        auto& nuEffFI = nuEffF.internalVector();

        NF_DEBUG_ASSERT(nuFI.size() == nutFI.size(), "nuF / nutF size mismatch");
        NF_DEBUG_ASSERT(nuEffFI.size() == nuFI.size(), "nuEffF size mismatch");

        const auto [nuVf, nutVf, nuEffVf] = views(nuFI, nutFI, nuEffFI);

        parallelFor(
            exec_,
            {0, nuEffFI.size()},
            NEON_LAMBDA(const localIdx f) { nuEffVf[f] = nuVf[f] + nutVf[f]; },
            "SA-DDES::correctNut::nuEffF"
        );
    }

    nuEffF.name = "nuEff";
}

void SpalartAllmarasDDES::calcNuTildeDiffusionCoeff(
    VolScalarField& nuTilde,
    const SurfScalarField& nuF,
    SurfScalarField& surfNuTilde,
    SurfScalarField& nuTildeEffF
) const
{
    nuTilde.correctBoundaryConditions();

    fvcc::SurfaceInterpolation<scalar> surfInterpol(
        exec_, nuTilde.mesh(), NeoN::TokenList({std::string("linear")})
    );

    surfInterpol.interpolate(nuTilde, surfNuTilde);

    const scalar invSigmaNut = scalar(1) / coeffs_.sigmaNut;

    const auto& nuFI = nuF.internalVector();
    const auto& nuTildeFI = surfNuTilde.internalVector();
    auto& nuTildeEffFI = nuTildeEffF.internalVector();

    const auto [nuVf, nuTildeVf, nuTildeEffVf] = views(nuFI, nuTildeFI, nuTildeEffFI);

    parallelFor(
        exec_,
        {0, nuTildeEffFI.size()},
        NEON_LAMBDA(const localIdx f) { nuTildeEffVf[f] = invSigmaNut * (nuVf[f] + nuTildeVf[f]); },
        "SA-DDES::calcNuTildeDiffusionCoeff"
    );

    nuTildeEffF.name = "nuTildeEff";
}

void SpalartAllmarasDDES::calcMagSqrVec(VolScalarField& magSqr, const VolVectorField& in) const
{

    const auto [value, magV] = views(in.internalVector(), magSqr.internalVector());

    parallelFor(
        exec_,
        {0, in.internalVector().size()},
        NEON_LAMBDA(const localIdx i) {
            magV[i] =
                value[i][0] * value[i][0] + value[i][1] * value[i][1] + value[i][2] * value[i][2];
        },
        "SA-DDES::magSqrGradNuTilde::internal"
    );
}

void SpalartAllmarasDDES::computeProdSpDDES(
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
) const
{
    const auto
        [nuTildeV,
         nuV,
         gxV,
         gyV,
         gzV,
         wallDistV,
         deltaV,
         gradNuTildeMagSqrV,
         productionV,
         spCoeffV] =
            views(
                nuTildeField.internalVector(),
                nuField.internalVector(),
                gradUx.internalVector(),
                gradUy.internalVector(),
                gradUz.internalVector(),
                wallDistanceField.internalVector(),
                deltaField.internalVector(),
                gradNuTildeMagSqrField.internalVector(),
                productionField.internalVector(),
                spCoeffField.internalVector()
            );

    // --- coefficients
    const scalar Cv1_3 = coeffs_.Cv1 * coeffs_.Cv1 * coeffs_.Cv1;
    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;

    const scalar Cb1 = coeffs_.Cb1;
    const scalar Cb2_s = coeffs_.Cb2 / coeffs_.sigmaNut;
    const scalar Cw1 = cw1_;
    const scalar Cw2 = coeffs_.Cw2;
    const scalar Cw3_6 = std::pow(coeffs_.Cw3, 6);

    const scalar fdCoef = coeffs_.fdCoef;
    const scalar Cdes = coeffs_.Cdes;
    const scalar Cs = coeffs_.Cs;
    const scalar fwStar = coeffs_.fwStar;

    const scalar ROOTVSMALL = scalar(1e-30);

    parallelFor(
        exec_,
        {0, nuTildeField.internalVector().size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar nuT = nuTildeV[i];
            const scalar nu = nuV[i];

            // =====================================================
            // chi, fv1, fv2
            // =====================================================
            const scalar chi = nuT / nu;
            const scalar chi2 = chi * chi;
            const scalar chi3 = chi2 * chi;

            const scalar fv1 = chi3 / (chi3 + Cv1_3);
            const scalar fv2 = scalar(1) - chi / (scalar(1) + chi * fv1);

            // =====================================================
            // magGradU (from gradU)
            // =====================================================
            const auto gx = gxV[i];
            const auto gy = gyV[i];
            const auto gz = gzV[i];

            const scalar magGradUSq = gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2] + gy[0] * gy[0]
                                    + gy[1] * gy[1] + gy[2] * gy[2] + gz[0] * gz[0] + gz[1] * gz[1]
                                    + gz[2] * gz[2];

            const scalar magGradU = std::sqrt(magGradUSq);

            // =====================================================
            // omega = 2*sqrt(a12² + a13² + a23²)
            // =====================================================
            const scalar a12 = scalar(0.5) * (gx[1] - gy[0]);
            const scalar a13 = scalar(0.5) * (gx[2] - gz[0]);
            const scalar a23 = scalar(0.5) * (gy[2] - gz[1]);

            const scalar omega = scalar(2.0) * std::sqrt(a12 * a12 + a13 * a13 + a23 * a23);

            // =====================================================
            // DDES shielding
            // =====================================================
            const scalar dWall = Kokkos::max(wallDistV[i], ROOTVSMALL);

            const scalar rD = Kokkos::min(
                (nu + nuT * fv1) / (kappa2 * dWall * dWall * Kokkos::max(magGradU, ROOTVSMALL)),
                scalar(10)
            );

            const scalar fD = scalar(1) - std::tanh(std::pow(fdCoef * rD, 3));

            // =====================================================
            // psi, dTilde
            // =====================================================
            const scalar psi = std::sqrt(Kokkos::min(
                scalar(100),
                (scalar(1) - Cb1 / (Cw1 * kappa2 * fwStar) * fv2) / Kokkos::max(fv1, ROOTVSMALL)
            ));

            const scalar dTilde = Kokkos::max(
                dWall - fD * Kokkos::max(dWall - psi * Cdes * deltaV[i], scalar(0)), ROOTVSMALL
            );

            const scalar invSqrdTilde = scalar(1) / (dTilde * dTilde);

            // =====================================================
            // sTilde (uses omega)
            // =====================================================
            const scalar sTilde =
                Kokkos::max(omega + fv2 * nuT * invSqrdTilde / kappa2, Cs * omega);

            // =====================================================
            // fw
            // =====================================================
            const scalar r = Kokkos::min(
                nuT / (Kokkos::max(sTilde, ROOTVSMALL) * kappa2 * dTilde * dTilde), scalar(10)
            );

            const scalar r6 = r * r * r * r * r * r;
            const scalar g = r + Cw2 * (r6 - r);
            const scalar g6 = g * g * g * g * g * g;

            const scalar fw = g * std::pow((scalar(1) + Cw3_6) / (g6 + Cw3_6), scalar(1.0 / 6.0));

            // =====================================================
            // Production + Sp
            // =====================================================
            productionV[i] = Cb1 * sTilde * nuT + Cb2_s * gradNuTildeMagSqrV[i];

            spCoeffV[i] = Cw1 * fw * nuT * invSqrdTilde;
        },
        "SA-DDES::prod+Sp+omega+magGradU/fused"
    );
}

} // namespace NeoN::turbulenceModels
