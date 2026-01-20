// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/turbulenceModels/DES/SpalartAllmarasBase.hpp"

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::turbulenceModels::DES
{

SpalartAllmarasBase::SpalartAllmarasBase(const Executor& exec, const UnstructuredMesh& mesh)
    : exec_(exec), mesh_(mesh),
      cw1_(coeffs_.Cb1 / (coeffs_.kappa * coeffs_.kappa) + (1.0 + coeffs_.Cb2) / coeffs_.sigmaNut)
{}

const SpalartAllmarasBase::Coefficients& SpalartAllmarasBase::coeffs() const { return coeffs_; }

scalar SpalartAllmarasBase::cw1() const { return cw1_; }

void SpalartAllmarasBase::omega(
    VolScalarField& omegaField,
    const VolVectorField& gradUx,
    const VolVectorField& gradUy,
    const VolVectorField& gradUz
) const
{
    // Internal
    const auto& gxI = gradUx.internalVector();
    const auto& gyI = gradUy.internalVector();
    const auto& gzI = gradUz.internalVector();
    auto& omI = omegaField.internalVector();

    NF_DEBUG_ASSERT(omI.size() == gxI.size(), "omega size mismatch.");
    NF_DEBUG_ASSERT(gyI.size() == gxI.size(), "omega gradUy size mismatch.");
    NF_DEBUG_ASSERT(gzI.size() == gxI.size(), "omega gradUz size mismatch.");

    // Boundary (value only)
    /*const auto& gxB = gradUx.boundaryData().value();
    const auto& gyB = gradUy.boundaryData().value();
    const auto& gzB = gradUz.boundaryData().value();
    auto& omB       = omegaField.boundaryData().value();

    // If you have calculated BCs everywhere, these sizes should match. If not, keep this assert.
    NF_DEBUG_ASSERT(omB.size() == gxB.size(), "omega boundary size mismatch.");
    NF_DEBUG_ASSERT(gyB.size() == gxB.size(), "omega boundary gradUy size mismatch.");
    NF_DEBUG_ASSERT(gzB.size() == gxB.size(), "omega boundary gradUz size mismatch.");
    */

    // ---- internal ----
    {
        const auto [gxV, gyV, gzV, omV] = views(gxI, gyI, gzI, omI);

        parallelFor(
            exec_,
            {0, omI.size()},
            NEON_LAMBDA(const localIdx i) {
                // gradU = [ gx; gy; gz ] in row form
                // gx = (dUx/dx, dUx/dy, dUx/dz)
                // gy = (dUy/dx, dUy/dy, dUy/dz)
                // gz = (dUz/dx, dUz/dy, dUz/dz)
                const auto gx = gxV[i];
                const auto gy = gyV[i];
                const auto gz = gzV[i];

                // skew(gradU) components:
                // a12 = 0.5*(dUx/dy - dUy/dx)
                // a13 = 0.5*(dUx/dz - dUz/dx)
                // a23 = 0.5*(dUy/dz - dUz/dy)
                const scalar a12 = scalar(0.5) * (gx[1] - gy[0]);
                const scalar a13 = scalar(0.5) * (gx[2] - gz[0]);
                const scalar a23 = scalar(0.5) * (gy[2] - gz[1]);

                // mag(skew(gradU)) = sqrt(2*(a12^2 + a13^2 + a23^2))
                // Omega = sqrt(2) * mag(skew(gradU)) = 2 * sqrt(a12^2 + a13^2 + a23^2)
                const scalar sum = a12 * a12 + a13 * a13 + a23 * a23;

                // Equivalent, slightly cheaper: Omega = 2*sqrt(sum)
                omV[i] = scalar(2.0) * std::sqrt(sum);
            },
            "SpalartAllmarasBase::omega/internal"
        );
    }
    /*
    // ---- boundary value ----
    {
        const auto [gxV, gyV, gzV, omV] = views(gxB, gyB, gzB, omB);

        parallelFor(
        exec_,
        {0, omB.size()},
        NEON_LAMBDA(const localIdx i)
        {
            const auto gx = gxV[i];
            const auto gy = gyV[i];
            const auto gz = gzV[i];

            const scalar a12 = scalar(0.5) * (gx[1] - gy[0]);
            const scalar a13 = scalar(0.5) * (gx[2] - gz[0]);
            const scalar a23 = scalar(0.5) * (gy[2] - gz[1]);

            const scalar sum = a12*a12 + a13*a13 + a23*a23;
            omV[i] = scalar(2.0) * std::sqrt(sum);
        },
        "SpalartAllmarasBase::omega/boundary"
        );
    }
    */
}

void SpalartAllmarasBase::magGradU(
    VolScalarField& magGradUField,
    const VolVectorField& gradUx,
    const VolVectorField& gradUy,
    const VolVectorField& gradUz
) const
{
    // -----------------
    // internal storage
    // -----------------
    const auto& gxI = gradUx.internalVector();
    const auto& gyI = gradUy.internalVector();
    const auto& gzI = gradUz.internalVector();
    auto& mgI = magGradUField.internalVector();

    NF_DEBUG_ASSERT(mgI.size() == gxI.size(), "magGradU size mismatch.");
    NF_DEBUG_ASSERT(gyI.size() == gxI.size(), "magGradU gradUy size mismatch.");
    NF_DEBUG_ASSERT(gzI.size() == gxI.size(), "magGradU gradUz size mismatch.");

    // -----------------
    // boundary values
    // -----------------
    const auto& gxB = gradUx.boundaryData().value();
    const auto& gyB = gradUy.boundaryData().value();
    const auto& gzB = gradUz.boundaryData().value();
    auto& mgB = magGradUField.boundaryData().value();

    NF_DEBUG_ASSERT(mgB.size() == gxB.size(), "magGradU boundary size mismatch.");
    NF_DEBUG_ASSERT(gyB.size() == gxB.size(), "magGradU boundary gradUy size mismatch.");
    NF_DEBUG_ASSERT(gzB.size() == gxB.size(), "magGradU boundary gradUz size mismatch.");

    // -----------------
    // internal kernel
    // -----------------
    {
        const auto [gxV, gyV, gzV, mgV] = views(gxI, gyI, gzI, mgI);

        parallelFor(
            exec_,
            {0, mgI.size()},
            NEON_LAMBDA(const localIdx i) {
                const auto gx = gxV[i];
                const auto gy = gyV[i];
                const auto gz = gzV[i];

                const scalar sum = gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2] + gy[0] * gy[0]
                                 + gy[1] * gy[1] + gy[2] * gy[2] + gz[0] * gz[0] + gz[1] * gz[1]
                                 + gz[2] * gz[2];

                mgV[i] = std::sqrt(sum);
            },
            "SpalartAllmarasBase::magGradU/internal"
        );
    }

    // -----------------
    // boundary kernel
    // -----------------
    {
        const auto [gxV, gyV, gzV, mgV] = views(gxB, gyB, gzB, mgB);

        parallelFor(
            exec_,
            {0, mgB.size()},
            NEON_LAMBDA(const localIdx i) {
                const auto gx = gxV[i];
                const auto gy = gyV[i];
                const auto gz = gzV[i];

                const scalar sum = gx[0] * gx[0] + gx[1] * gx[1] + gx[2] * gx[2] + gy[0] * gy[0]
                                 + gy[1] * gy[1] + gy[2] * gy[2] + gz[0] * gz[0] + gz[1] * gz[1]
                                 + gz[2] * gz[2];

                mgV[i] = std::sqrt(sum);
            },
            "SpalartAllmarasBase::magGradU/boundary"
        );
    }
}

void SpalartAllmarasBase::chi(
    VolScalarField& chiField, const VolScalarField& nuTilde, const VolScalarField& nu
) const
{
    // --- Internal data
    const auto& nuTildeInt = nuTilde.internalVector();
    const auto& nuInt = nu.internalVector();
    auto& chiInt = chiField.internalVector();

    NF_DEBUG_ASSERT(chiInt.size() == nuTildeInt.size(), "chi internal size mismatch.");
    NF_DEBUG_ASSERT(nuInt.size() == nuTildeInt.size(), "nu internal size mismatch.");

    // --- Boundary data (value only; algebraic field)
    /*    const auto& nuTildeBnd = nuTilde.boundaryData().value();
        const auto& nuBnd     = nu.boundaryData().value();
        auto&       chiBnd    = chiField.boundaryData().value();

        NF_DEBUG_ASSERT(chiBnd.size()    == nuTildeBnd.size(), "chi boundary size mismatch.");
        NF_DEBUG_ASSERT(nuBnd.size()     == nuTildeBnd.size(), "nu boundary size mismatch.");
    */
    // --- Create views
    const auto [nuTildeIntV, nuIntV, chiIntV] = views(nuTildeInt, nuInt, chiInt);

    // const auto [nuTildeBndV, nuBndV, chiBndV] = views(nuTildeBnd, nuBnd, chiBnd);

    // --- Internal computation
    parallelFor(
        exec_,
        {0, chiInt.size()},
        NEON_LAMBDA(const localIdx i) { chiIntV[i] = nuTildeIntV[i] / nuIntV[i]; },
        "SpalartAllmarasBase::chi::internal"
    );

    // --- Boundary computation
    /*    parallelFor(
            exec_,
            {0, chiBnd.size()},
            NEON_LAMBDA(const localIdx i) {
                chiBndV[i] = nuTildeBndV[i] / nuBndV[i];
            },
            "SpalartAllmarasBase::chi::boundary"
        );
    */
}

void SpalartAllmarasBase::fv1(VolScalarField& fv1Field, const VolScalarField& chiField) const
{
    const auto& chiI = chiField.internalVector();
    auto& fv1I = fv1Field.internalVector();

    // const auto& chiB = chiField.boundaryData().value();
    // auto& fv1B       = fv1Field.boundaryData().value();

    NF_DEBUG_ASSERT(fv1I.size() == chiI.size(), "fv1 internal size mismatch.");
    // NF_DEBUG_ASSERT(fv1B.size() == chiB.size(), "fv1 boundary size mismatch.");

    const scalar cv1 = coeffs_.Cv1;
    const scalar cv1Cubed = cv1 * cv1 * cv1;

    {
        const auto [chiV, fv1V] = views(chiI, fv1I);
        parallelFor(
            exec_,
            {0, fv1Field.size()},
            NEON_LAMBDA(const localIdx i) {
                const scalar chi = chiV[i];
                const scalar chi3 = chi * chi * chi;
                fv1V[i] = chi3 / (chi3 + cv1Cubed);
            },
            "SA::fv1/internal"
        );
    }
    /*
    {
        const auto [chiV, fv1V] = views(chiB, fv1B);
        parallelFor(
        exec_,
        {0, fv1B.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar chi = chiV[i];
            const scalar chi3 = chi * chi * chi;
            fv1V[i] = chi3 / (chi3 + cv1Cubed);
        },
        "SA::fv1/boundary"
        );
    }
    */
}

void SpalartAllmarasBase::fv2(
    VolScalarField& fv2Field, const VolScalarField& chiField, const VolScalarField& fv1Field
) const
{
    const auto& chiI = chiField.internalVector();
    const auto& fv1I = fv1Field.internalVector();
    auto& fv2I = fv2Field.internalVector();

    NF_DEBUG_ASSERT(fv2I.size() == chiI.size(), "fv2 internal size mismatch.");

    {
        const auto [chiV, fv1V, fv2V] = views(chiI, fv1I, fv2I);
        parallelFor(
            exec_,
            {0, fv2Field.size()},
            NEON_LAMBDA(const localIdx i) {
                const scalar chi = chiV[i];
                fv2V[i] = 1.0 - chi / (1.0 + chi * fv1V[i]);
            },
            "SA::fv2/internal"
        );
    }
}

void SpalartAllmarasBase::stilda(
    VolScalarField& stildaField,
    const VolScalarField& omega,
    const VolScalarField& nuTilde,
    const VolScalarField& dTilde,
    const VolScalarField& fv2Field
) const
{
    const auto& omI = omega.internalVector();
    const auto& nuI = nuTilde.internalVector();
    const auto& dI = dTilde.internalVector();
    const auto& fv2I = fv2Field.internalVector();
    auto& stI = stildaField.internalVector();

    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar Cs = coeffs_.Cs;

    {
        const auto [omV, nuV, dV, fv2V, stV] = views(omI, nuI, dI, fv2I, stI);
        parallelFor(
            exec_,
            {0, stildaField.size()},
            NEON_LAMBDA(const localIdx i) {
                const scalar denom = kappa2 * dV[i] * dV[i] + ROOTVSMALL;
                stV[i] = Kokkos::max(omV[i] + fv2V[i] * nuV[i] / denom, Cs * omV[i]);
            },
            "SA::stilda/internal"
        );
    }
}

void SpalartAllmarasBase::fw(
    VolScalarField& fwField,
    const VolScalarField& stildaField,
    const VolScalarField& dTilde,
    const VolScalarField& nuTilde
) const
{
    const auto& stI = stildaField.internalVector();
    const auto& dI = dTilde.internalVector();
    const auto& nuI = nuTilde.internalVector();
    auto& fwI = fwField.internalVector();

    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;
    const scalar cw2 = coeffs_.Cw2;
    const scalar cw3 = coeffs_.Cw3;
    const scalar cw3Pow6 = std::pow(cw3, 6.0);

    {
        const auto [stV, dV, nuV, fwV] = views(stI, dI, nuI, fwI);
        parallelFor(
            exec_,
            {0, fwField.size()},
            NEON_LAMBDA(const localIdx i) {
                const scalar denom = stV[i] * kappa2 * dV[i] * dV[i] + ROOTVSMALL;
                const scalar r = std::min(scalar(10), nuV[i] / denom);
                const scalar g = r + cw2 * (std::pow(r, 6.0) - r);
                const scalar g6 = std::pow(g, 6.0);
                fwV[i] = g * std::pow((1.0 + cw3Pow6) / (g6 + cw3Pow6), scalar(1.0 / 6.0));
            },
            "SA::fw/internal"
        );
    }
}

void SpalartAllmarasBase::nut(
    VolScalarField& nutField, const VolScalarField& nuTilde, const VolScalarField& fv1Field
) const
{
    const auto& nuI = nuTilde.internalVector();
    const auto& fv1I = fv1Field.internalVector();
    auto& nutI = nutField.internalVector();

    //    const auto& nuB = nuTilde.boundaryData().value();
    //    const auto& fv1B = fv1Field.boundaryData().value();
    //    auto& nutB = nutField.boundaryData().value();

    {
        const auto [nuV, fv1V, nutV] = views(nuI, fv1I, nutI);
        parallelFor(
            exec_,
            {0, nutField.size()},
            NEON_LAMBDA(const localIdx i) { nutV[i] = nuV[i] * fv1V[i]; },
            "SA::nut/internal"
        );
    }
    /*
    {
        const auto [nuV, fv1V, nutV] = views(nuB, fv1B, nutB);
        parallelFor(
        exec_,
        {0, nutB.size()},
        NEON_LAMBDA(const localIdx i) {
            nutV[i] = nuV[i] * fv1V[i];
        },
        "SA::nut/boundary"
        );
    }
    */
}

void SpalartAllmarasBase::computeProdSpDDES(
    VolScalarField& productionField,
    VolScalarField& spCoeffField,
    const VolScalarField& nuTildeField,
    const VolScalarField& nuField,
    const VolScalarField& omegaField,
    const VolScalarField& wallDistanceField,
    const VolScalarField& magGradUField,
    const VolScalarField& deltaField,
    const VolScalarField& gradNuTildeMagSqrField
) const
{
    const auto
        [nuTildeV,
         nuV,
         omegaV,
         wallDistV,
         magGradUV,
         deltaV,
         gradNuTildeMagSqrV,
         productionV,
         spCoeffV] =
            views(
                nuTildeField.internalVector(),
                nuField.internalVector(),
                omegaField.internalVector(),
                wallDistanceField.internalVector(),
                magGradUField.internalVector(),
                deltaField.internalVector(),
                gradNuTildeMagSqrField.internalVector(),
                productionField.internalVector(),
                spCoeffField.internalVector()
            );

    // --- coefficients (unchanged)
    const scalar Cv1_3 = std::pow(coeffs_.Cv1, 3);
    const scalar kappa2 = coeffs_.kappa * coeffs_.kappa;

    const scalar Cb1 = coeffs_.Cb1;
    const scalar Cb2_s = coeffs_.Cb2 / coeffs_.sigmaNut;
    const scalar Cw1 = cw1_;
    const scalar Cw2 = coeffs_.Cw2;
    const scalar Cw3_6 = std::pow(coeffs_.Cw3, 6);

    const scalar fdCoef = coeffs_.fdCoef;
    const scalar Cdes = coeffs_.Cdes;
    const scalar Cs = coeffs_.Cs;
    const scalar Ct3 = coeffs_.Ct3;
    const scalar ROOTVSMALL(1e-30);

    parallelFor(
        exec_,
        {0, nuTildeField.internalVector().size()},
        NEON_LAMBDA(const localIdx i) {
            // -------------------------
            // chi, fv1, fv2
            // -------------------------
            const scalar chi = nuTildeV[i] / nuV[i];
            const scalar chi3 = chi * chi * chi;

            const scalar fv1 = chi3 / (chi3 + Cv1_3);

            const scalar fv2 = scalar(1) - chi / (scalar(1) + chi * fv1);

            // -------------------------
            // DDES shielding (uses magGradU)
            // -------------------------
            const scalar dWall = Kokkos::max(wallDistV[i], ROOTVSMALL);

            const scalar rD = Kokkos::min(
                (nuV[i] + nuTildeV[i])
                    / (kappa2 * dWall * dWall * Kokkos::max(magGradUV[i], ROOTVSMALL)),
                scalar(10)
            );

            const scalar fD = scalar(1) - std::tanh(std::pow(fdCoef * rD, 3));

            // -------------------------
            // psi (DES limiter)
            // -------------------------
            const scalar psi = std::sqrt(Kokkos::min(
                scalar(100),
                (scalar(1) - Cb1 / (Cw1 * kappa2 * Ct3) * fv2) / Kokkos::max(fv1, ROOTVSMALL)
            ));

            // -------------------------
            // dTilde and inverse square
            // -------------------------
            const scalar dTilde = Kokkos::max(
                dWall - fD * Kokkos::max(dWall - psi * Cdes * deltaV[i], scalar(0)), ROOTVSMALL
            );

            const scalar invSqrdTilde = scalar(1) / (dTilde * dTilde);

            // -------------------------
            // sTilde (uses omega)
            // -------------------------
            const scalar sTilde =
                Kokkos::max(omegaV[i] + fv2 * nuTildeV[i] * invSqrdTilde / kappa2, Cs * omegaV[i]);

            // -------------------------
            // fw
            // -------------------------
            const scalar r = Kokkos::min(
                nuTildeV[i] / (sTilde * kappa2 * invSqrdTilde + ROOTVSMALL), scalar(10)
            );

            const scalar r6 = std::pow(r, 6);
            const scalar g = r + Cw2 * (r6 - r);
            const scalar g6 = std::pow(g, 6);

            const scalar fw = g * std::pow((scalar(1) + Cw3_6) / (g6 + Cw3_6), scalar(1.0 / 6.0));

            // -------------------------
            // production (explicit)
            // -------------------------
            productionV[i] = Cb1 * sTilde * nuTildeV[i] + Cb2_s * gradNuTildeMagSqrV[i];

            // -------------------------
            // Sp coefficient (implicit)
            // -------------------------
            spCoeffV[i] = Cw1 * fw * nuTildeV[i] * invSqrdTilde;
        },
        "SA-DDES::nut+prod+Sp+dTilde/fused"
    );
}

} // namespace NeoN::turbulenceModels::DES
