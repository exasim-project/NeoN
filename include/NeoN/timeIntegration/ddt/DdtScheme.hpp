// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/ddtFluxCorrBoundary.hpp"

namespace NeoN
{
namespace timeIntegration
{

/**
 * @brief Time-derivative discretisation policy.
 * Provides coefficients for implicit ddt assembly.
 */
class DdtScheme
{
public:

    using VolVectorField = NeoN::finiteVolume::cellCentred::VolumeField<Vec3>;
    using SurfScalarField = NeoN::finiteVolume::cellCentred::SurfaceField<scalar>;

    virtual ~DdtScheme() = default;

    /// Number of historical time levels required
    virtual int nSteps() const = 0;

    /**
     * @brief Coefficient multiplying phi^{n+1}
     */
    virtual scalar a0(scalar dt) const = 0;

    /**
     * @brief Coefficient multiplying phi^{n}
     */
    virtual scalar a1(scalar dt) const = 0;

    /**
     * @brief Coefficient multiplying phi^{n-1}
     * default: unused
     */
    virtual scalar a2(scalar) const { return scalar(0); }

    /**
     * @brief Coefficients for schemes using multiple time levels
     * at the first timestep (startup phase) when the old time level phi^{n-1} is
     * not present yet
     */
    virtual scalar a0Startup(scalar dt) const { return a0(dt); }
    virtual scalar a1Startup(scalar dt) const { return a1(dt); }

    SurfScalarField
    ddtFluxCorr(const VolVectorField& u, const SurfScalarField& flux, scalar dt) const
    {
        const auto& mesh = u.mesh();
        const auto exec = flux.exec();

        // --- interpolation (same for all schemes)
        NeoN::finiteVolume::cellCentred::SurfaceInterpolation<Vec3> interp(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );

        // --- boundary conditions consistent with U
        auto surfaceBCs = NeoN::finiteVolume::cellCentred::createFluxCorrBCsFromU(mesh, u);

        SurfScalarField fluxCorr(exec, std::string("ddtFluxCorr"), mesh, surfaceBCs);

        auto [outV, sfV] = views(fluxCorr.internalVector(), mesh.faceAreas());

        const size_t n = outV.size();

        // ===============================
        // BDF1 contribution (phi^{n})
        // ===============================
        const auto& u0 = NeoN::finiteVolume::cellCentred::oldTime(u);
        const auto& flux0 = NeoN::finiteVolume::cellCentred::oldTime(flux);

        auto uf0 = interp.interpolate(u0);

        auto [flux0V, uf0V] = views(flux0.internalVector(), uf0.internalVector());

        const scalar w1 = a1(dt);

        NeoN::parallelFor(
            exec,
            {size_t(0), n},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto d = (sfV[i] & uf0V[i]);
                const auto corr = flux0V[i] - d;

                const scalar limiter = ddtFluxCorrLimiter(mag(flux0V[i]), mag(corr));

                outV[i] = limiter * w1 * corr;
            },
            "ddtFluxCorr::BDF1"
        );

        // ===============================
        // BDF2 contribution (phi^{n-1})
        // ===============================
        if (nSteps() >= 2)
        {
            const auto& u00 =
                NeoN::finiteVolume::cellCentred::oldTime(NeoN::finiteVolume::cellCentred::oldTime(u)
                );
            const auto& flux00 = NeoN::finiteVolume::cellCentred::oldTime(
                NeoN::finiteVolume::cellCentred::oldTime(flux)
            );

            auto uf00 = interp.interpolate(u00);

            auto [flux00V, uf00V] = views(flux00.internalVector(), uf00.internalVector());

            const scalar w2 = a2(dt);

            NeoN::parallelFor(
                exec,
                {size_t(0), n},
                KOKKOS_LAMBDA(const localIdx i) {
                    const auto d = (sfV[i] & uf00V[i]);
                    const auto corr = flux00V[i] - d;

                    const scalar limiter = ddtFluxCorrLimiter(mag(flux00V[i]), mag(corr));

                    outV[i] += limiter * w2 * corr;
                },
                "ddtFluxCorr::BDF2"
            );
        }

        fluxCorr.correctBoundaryConditions();
        return fluxCorr;
    }

protected:

    // ------------------------------------------------------------------ //
    // OpenFOAM-compatible limiter (with numerical safety)
    // ------------------------------------------------------------------ //
    KOKKOS_INLINE_FUNCTION
    static scalar ddtFluxCorrLimiter(const scalar fluxMag, const scalar corrMag)
    {
        // OpenFOAM SMALL (double precision)
        constexpr scalar small = scalar(1e-30);

        const scalar ratio = corrMag / (fluxMag + small);
        return scalar(1.0) - Kokkos::min(ratio, scalar(1.0));
    }
};

} // namespace timeIntegration
} // namespace NeoN
