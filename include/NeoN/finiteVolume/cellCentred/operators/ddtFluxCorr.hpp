// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/ddtFluxCorrBoundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

using VolVectorField = VolumeField<Vec3>;
using SurfScalarField = SurfaceField<scalar>;

namespace detail
{

KOKKOS_INLINE_FUNCTION
scalar ddtFluxCorrLimiter(const scalar fluxMag, const scalar corrMag)
{
    constexpr scalar small = 1.0e-30;
    const auto ratio = corrMag / (fluxMag + small);
    return scalar(1) - Kokkos::min(ratio, scalar(1));
}

// ------------------------------------------------------------------
// BDF1 kernel (also used for BDF2 startup)
// ------------------------------------------------------------------
inline void ddtFluxCorrBDF1Kernel(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    const SurfScalarField& flux0,
    const SurfaceField<Vec3>& uf0,
    SurfScalarField& fluxCorr,
    scalar dt
)
{
    const scalar a1 = scalar(1) / dt;
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Internal faces
    auto [outV, flux0V, uf0V, SfV] = views(
        fluxCorr.internalVector(), flux0.internalVector(), uf0.internalVector(), mesh.faceNormals()
    );
    parallelFor(
        exec,
        {size_t(0), static_cast<size_t>(nInternalFaces)},
        NEON_LAMBDA(const localIdx i) {
            const auto d = (SfV[i] & uf0V[i]);
            const auto corr = flux0V[i] - d;
            const scalar limiter = ddtFluxCorrLimiter(mag(flux0V[i]), mag(corr));
            outV[i] = limiter * a1 * corr;
        },
        "ddtFluxCorr::BDF1::internal"
    );

    // Boundary faces
    auto [outBV, flux0BV, uf0BV] = views(
        fluxCorr.boundaryData().value(), flux0.boundaryData().value(), uf0.boundaryData().value()
    );
    parallelFor(
        exec,
        {size_t(0), static_cast<size_t>(nBoundaryFaces)},
        NEON_LAMBDA(const localIdx bfi) {
            const auto d = (SfV[nInternalFaces + bfi] & uf0BV[bfi]);
            const auto corr = flux0BV[bfi] - d;
            const scalar limiter = ddtFluxCorrLimiter(mag(flux0BV[bfi]), mag(corr));
            outBV[bfi] = limiter * a1 * corr;
        },
        "ddtFluxCorr::BDF1::boundary"
    );
}

// ------------------------------------------------------------------
// BDF2 kernel
// ------------------------------------------------------------------
inline void ddtFluxCorrBDF2Kernel(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    const SurfScalarField& flux0,
    const SurfScalarField& flux00,
    const SurfaceField<Vec3>& uf0,
    const SurfaceField<Vec3>& uf00,
    SurfScalarField& fluxCorr,
    scalar dt
)
{
    const scalar a1 = 2.0 / dt;
    const scalar a2 = -0.5 / dt;
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    // Internal faces
    {
        auto [outV, flux0V, flux00V, uf0V, uf00V, SfV] = views(
            fluxCorr.internalVector(),
            flux0.internalVector(),
            flux00.internalVector(),
            uf0.internalVector(),
            uf00.internalVector(),
            mesh.faceNormals()
        );
        parallelFor(
            exec,
            {size_t(0), static_cast<size_t>(nInternalFaces)},
            NEON_LAMBDA(const localIdx i) {
                const auto d1 = (SfV[i] & uf0V[i]);
                const auto corr1 = flux0V[i] - d1;

                const auto d2 = (SfV[i] & uf00V[i]);
                const auto corr2 = flux00V[i] - d2;

                const scalar limiter1 = ddtFluxCorrLimiter(mag(flux0V[i]), mag(corr1));
                const scalar limiter2 = ddtFluxCorrLimiter(mag(flux00V[i]), mag(corr2));

                outV[i] = limiter1 * a1 * corr1 + limiter2 * a2 * corr2;
            },
            "ddtFluxCorr::BDF2::internal"
        );
    }

    // Boundary faces
    {
        auto outBV = fluxCorr.boundaryData().value().view();
        auto flux0BV = flux0.boundaryData().value().view();
        auto flux00BV = flux00.boundaryData().value().view();
        auto uf0BV = uf0.boundaryData().value().view();
        auto uf00BV = uf00.boundaryData().value().view();
        auto SfV = mesh.faceNormals().view();
        parallelFor(
            exec,
            {size_t(0), static_cast<size_t>(nBoundaryFaces)},
            NEON_LAMBDA(const localIdx bfi) {
                const auto d1 = (SfV[nInternalFaces + bfi] & uf0BV[bfi]);
                const auto corr1 = flux0BV[bfi] - d1;

                const auto d2 = (SfV[nInternalFaces + bfi] & uf00BV[bfi]);
                const auto corr2 = flux00BV[bfi] - d2;

                const scalar limiter1 = ddtFluxCorrLimiter(mag(flux0BV[bfi]), mag(corr1));
                const scalar limiter2 = ddtFluxCorrLimiter(mag(flux00BV[bfi]), mag(corr2));

                outBV[bfi] = limiter1 * a1 * corr1 + limiter2 * a2 * corr2;
            },
            "ddtFluxCorr::BDF2::boundary"
        );
    }
}

} // namespace detail

inline SurfScalarField
ddtFluxCorr(const VolVectorField& u, const SurfScalarField& phi, scalar dt, DdtScheme scheme)
{
    const auto& mesh = u.mesh();
    const auto& exec = phi.exec();

    // --- interpolation
    SurfaceInterpolation<Vec3> interp(exec, mesh, TokenList({std::string("linear")}));

    // --- boundary conditions consistent with U
    auto surfaceBCs = createFluxCorrBCsFromU(mesh, u);

    SurfScalarField fluxCorr(exec, "ddtFluxCorr", mesh, surfaceBCs);

    const int level = oldTimeLevel(u);

    // --- BDF1 / startup
    const auto& u0 = oldTime(u);
    const auto& phi0 = oldTime(phi);
    auto uf0 = interp.interpolate(u0);

    if (scheme == DdtScheme::BDF2 && level >= 2) // --- BDF2 contribution
    {
        const auto& u00 = oldTime(u0);
        const auto& phi00 = oldTime(phi0);
        auto uf00 = interp.interpolate(u00);

        detail::ddtFluxCorrBDF2Kernel(exec, mesh, phi0, phi00, uf0, uf00, fluxCorr, dt);
    }
    else
    {
        detail::ddtFluxCorrBDF1Kernel(exec, mesh, phi0, uf0, fluxCorr, dt);
    }

    fluxCorr.correctBoundaryConditions();
    return fluxCorr;
}

} // namespace NeoN::finiteVolume::cellCentred
