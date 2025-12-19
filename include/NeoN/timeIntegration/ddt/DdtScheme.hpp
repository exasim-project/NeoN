// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surface/ddtPhiCorrBoundary.hpp"

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

    SurfScalarField ddtPhiCorr(const VolVectorField& U, const SurfScalarField& phi, scalar dt) const
    {
        const auto& mesh = U.mesh();
        const auto exec = phi.exec();

        // --- interpolation (same for all schemes)
        NeoN::finiteVolume::cellCentred::SurfaceInterpolation<Vec3> interp(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );

        // --- boundary conditions consistent with U
        auto surfaceBCs = NeoN::finiteVolume::cellCentred::createPhiCorrBCsFromU(mesh, U);

        SurfScalarField phiCorr(exec, std::string("ddtPhiCorr"), mesh, surfaceBCs);

        auto [outV, SfV] = views(phiCorr.internalVector(), mesh.faceAreas());

        const size_t N = outV.size();

        // ===============================
        // BDF1 contribution (phi^{n})
        // ===============================
        const auto& U0 = NeoN::finiteVolume::cellCentred::oldTime(U);
        const auto& phi0 = NeoN::finiteVolume::cellCentred::oldTime(phi);

        auto Uf0 = interp.interpolate(U0);

        auto [phi0V, Uf0V] = views(phi0.internalVector(), Uf0.internalVector());

        const scalar w1 = a1(dt);

        NeoN::parallelFor(
            exec,
            {size_t(0), N},
            KOKKOS_LAMBDA(const localIdx i) {
                const auto d = (SfV[i] & Uf0V[i]);
                const auto corr = phi0V[i] - d;

                const scalar limiter = ddtPhiCorrLimiter(mag(phi0V[i]), mag(corr));

                outV[i] += limiter * w1 * corr;
            },
            "ddtPhiCorr::BDF1"
        );

        // ===============================
        // BDF2 contribution (phi^{n-1})
        // ===============================
        if (nSteps() >= 2)
        {
            const auto& U00 =
                NeoN::finiteVolume::cellCentred::oldTime(NeoN::finiteVolume::cellCentred::oldTime(U)
                );
            const auto& phi00 = NeoN::finiteVolume::cellCentred::oldTime(
                NeoN::finiteVolume::cellCentred::oldTime(phi)
            );

            auto Uf00 = interp.interpolate(U00);

            auto [phi00V, Uf00V] = views(phi00.internalVector(), Uf00.internalVector());

            const scalar w2 = a2(dt);

            NeoN::parallelFor(
                exec,
                {size_t(0), N},
                KOKKOS_LAMBDA(const localIdx i) {
                    const auto d = (SfV[i] & Uf00V[i]);
                    const auto corr = phi00V[i] - d;

                    const scalar limiter = ddtPhiCorrLimiter(mag(phi00V[i]), mag(corr));

                    outV[i] += limiter * w2 * corr;
                },
                "ddtPhiCorr::BDF2"
            );
        }

        phiCorr.correctBoundaryConditions();
        return phiCorr;
    }

protected:

    // ------------------------------------------------------------------ //
    // OpenFOAM-compatible limiter (with numerical safety)
    // ------------------------------------------------------------------ //
    KOKKOS_INLINE_FUNCTION
    static scalar ddtPhiCorrLimiter(const scalar phiMag, const scalar corrMag)
    {
        // OpenFOAM SMALL (double precision)
        constexpr scalar SMALL = scalar(1e-30);

        const scalar ratio = corrMag / (phiMag + SMALL);
        return scalar(1.0) - Kokkos::min(ratio, scalar(1.0));
    }
};

} // namespace timeIntegration
} // namespace NeoN
