// SPDX-FileCopyrightText: 2011-2017 OpenFOAM Foundation
// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT
//
// Transcription of Foam::MULES::explicitSolve / limit / limiter
// (OpenFOAM src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C) for the
// simplified VoF path: rho == 1, Sp == Su == 0, static mesh,
// extremaCoeff == smoothLimiter == 0, no coupled/fixed-value boundary widening.

#include "NeoN/finiteVolume/cellCentred/operators/mules.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/vector/vector.hpp"

namespace NeoN::finiteVolume::cellCentred
{

void mulesExplicitSolve(
    VolumeField<scalar>& alpha,
    const SurfaceField<scalar>& phi,
    SurfaceField<scalar>& alphaPhi,
    scalar deltaT,
    scalar psiMax,
    scalar psiMin,
    int nLimiterIter
)
{
    const auto& mesh = alpha.mesh();
    const auto exec = alpha.exec();
    const auto nCells = mesh.nCells();
    const auto nInt = mesh.nInternalFaces();
    const auto nBnd = mesh.nBoundaryFaces();
    const scalar rDeltaT = 1.0 / deltaT;
    // OpenFOAM's ROOTVSMALL (sqrt of the smallest normalised double).
    const scalar ROOTVSMALL = 1e-18;

    const auto own = mesh.faceOwners().view();
    const auto nei = mesh.faceNeighbors().view();
    const auto V = mesh.cellVolumes().view();
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();

    auto a = alpha.internalVector().view();     // current == oldTime (psi0) at entry
    auto ap = alphaPhi.internalVector().view(); // high-order flux in; limited out
    const auto phiI = phi.internalVector().view();
    const auto apB = alphaPhi.boundaryData().value().view(); // boundary alphaPhi

    // --- donor (upwind) flux phiBD + correction phiCorr on internal faces ---
    // boundary: phiBD == alphaPhi (⇒ phiCorr == 0), handled implicitly below.
    Vector<scalar> PhiBD(exec, nInt, 0.0), PhiCorr(exec, nInt, 0.0);
    auto phiBD = PhiBD.view();
    auto phiCorr = PhiCorr.view();
    parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) {
            const scalar donor = (phiI[f] >= 0.0) ? a[own[f]] : a[nei[f]];
            phiBD[f] = donor * phiI[f];
            phiCorr[f] = ap[f] - phiBD[f];
        },
        "mules::phiBD"
    );

    // --- limiter: per-cell allowable extrema and bound RHS (MULESTemplates.C limiter) ---
    Vector<scalar> PsiMaxn(exec, nCells, psiMin), PsiMinn(exec, nCells, psiMax);
    Vector<scalar> SumPhiBD(exec, nCells, 0.0), SumPhip(exec, nCells, 0.0),
        MSumPhim(exec, nCells, 0.0);
    auto psiMaxn = PsiMaxn.view();
    auto psiMinn = PsiMinn.view();
    auto sumPhiBD = SumPhiBD.view();
    auto sumPhip = SumPhip.view();
    auto mSumPhim = MSumPhim.view();

    parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) {
            const localIdx o = own[f];
            const localIdx n = nei[f];
            Kokkos::atomic_max(&psiMaxn[o], a[n]);
            Kokkos::atomic_min(&psiMinn[o], a[n]);
            Kokkos::atomic_max(&psiMaxn[n], a[o]);
            Kokkos::atomic_min(&psiMinn[n], a[o]);

            Kokkos::atomic_add(&sumPhiBD[o], phiBD[f]);
            Kokkos::atomic_sub(&sumPhiBD[n], phiBD[f]);

            const scalar pc = phiCorr[f];
            if (pc > 0.0)
            {
                Kokkos::atomic_add(&sumPhip[o], pc);
                Kokkos::atomic_add(&mSumPhim[n], pc);
            }
            else
            {
                Kokkos::atomic_sub(&mSumPhim[o], pc);
                Kokkos::atomic_sub(&sumPhip[n], pc);
            }
        },
        "mules::extremaInternal"
    );

    // boundary faces (non-coupled): phiCorr_b == 0, so only sumPhiBD accumulates.
    parallelFor(
        exec, {0, nBnd},
        KOKKOS_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&sumPhiBD[bOwn[bf]], apB[bf]); },
        "mules::extremaBoundary"
    );

    // extremaCoeff == 0, smoothLimiter == 0, static mesh, rho == 1, Sp == Su == 0:
    //   psiMaxn = min(psiMaxn, psiMax);  psiMinn = max(psiMinn, psiMin);
    //   psiMaxn = V*rDeltaT*(psiMaxn - psi0) + sumPhiBD
    //   psiMinn = V*rDeltaT*(psi0 - psiMinn) - sumPhiBD
    parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) {
            scalar mx = Kokkos::min(psiMaxn[c], psiMax);
            scalar mn = Kokkos::max(psiMinn[c], psiMin);
            const scalar vr = V[c] * rDeltaT;
            psiMaxn[c] = vr * (mx - a[c]) + sumPhiBD[c];
            psiMinn[c] = vr * (a[c] - mn) - sumPhiBD[c];
        },
        "mules::boundRHS"
    );

    // --- FCT limiter sweeps ---
    Vector<scalar> Lambda(exec, nInt, 1.0);
    auto lambda = Lambda.view();
    Vector<scalar> SumlPhip(exec, nCells, 0.0), MSumlPhim(exec, nCells, 0.0);
    auto sumlPhip = SumlPhip.view();
    auto mSumlPhim = MSumlPhim.view();

    for (int j = 0; j < nLimiterIter; ++j)
    {
        parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) {
                sumlPhip[c] = 0.0;
                mSumlPhim[c] = 0.0;
            },
            "mules::zeroSuml"
        );

        parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                const scalar lpc = lambda[f] * phiCorr[f];
                if (lpc > 0.0)
                {
                    Kokkos::atomic_add(&sumlPhip[own[f]], lpc);
                    Kokkos::atomic_add(&mSumlPhim[nei[f]], lpc);
                }
                else
                {
                    Kokkos::atomic_sub(&mSumlPhim[own[f]], lpc);
                    Kokkos::atomic_sub(&sumlPhip[nei[f]], lpc);
                }
            },
            "mules::sumlInternal"
        );

        // per-cell limiter coefficients (note OpenFOAM's naming swap:
        //   lambdam == clamped sumlPhip,  lambdap == clamped mSumlPhim).
        parallelFor(
            exec, {0, nCells},
            KOKKOS_LAMBDA(const localIdx c) {
                sumlPhip[c] = Kokkos::max(
                    Kokkos::min((sumlPhip[c] + psiMaxn[c]) / (mSumPhim[c] + ROOTVSMALL), 1.0), 0.0
                );
                mSumlPhim[c] = Kokkos::max(
                    Kokkos::min((mSumlPhim[c] + psiMinn[c]) / (sumPhip[c] + ROOTVSMALL), 1.0), 0.0
                );
            },
            "mules::lambdaCells"
        );
        // lambdam := sumlPhip, lambdap := mSumlPhim
        auto lambdam = sumlPhip;
        auto lambdap = mSumlPhim;

        parallelFor(
            exec, {0, nInt},
            KOKKOS_LAMBDA(const localIdx f) {
                if (phiCorr[f] > 0.0)
                {
                    lambda[f] = Kokkos::min(
                        lambda[f], Kokkos::min(lambdap[own[f]], lambdam[nei[f]])
                    );
                }
                else
                {
                    lambda[f] = Kokkos::min(
                        lambda[f], Kokkos::min(lambdam[own[f]], lambdap[nei[f]])
                    );
                }
            },
            "mules::lambdaFaces"
        );
    }

    // --- apply limiter: alphaPhi = phiBD + lambda*phiCorr (internal; boundary unchanged) ---
    parallelFor(
        exec, {0, nInt},
        KOKKOS_LAMBDA(const localIdx f) { ap[f] = phiBD[f] + lambda[f] * phiCorr[f]; },
        "mules::applyLimiter"
    );

    // --- conservative explicit update: alpha = psi0 - deltaT*surfaceIntegrate(alphaPhi) ---
    Vector<scalar> Div(exec, nCells, 0.0);
    surfaceIntegrate<scalar>(
        exec, nInt, nei, own, bOwn, alphaPhi.internalVector().view(), apB, V, Div.view(),
        dsl::Coeff(1.0)
    );
    auto div = Div.view();
    parallelFor(
        exec, {0, nCells},
        KOKKOS_LAMBDA(const localIdx c) { a[c] = a[c] - deltaT * div[c]; },
        "mules::update"
    );

    alpha.correctBoundaryConditions();
}

} // namespace NeoN::finiteVolume::cellCentred
