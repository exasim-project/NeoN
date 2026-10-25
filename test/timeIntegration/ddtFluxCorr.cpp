// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "../dsl/common.hpp"
#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using Scalar = NeoN::scalar;
using NeoN::sizeType;
using NeoN::toLocalIdx;
using Vec3 = NeoN::Vec3;
using VolScalar = fvcc::VolumeField<Scalar>;
using VolVector = fvcc::VolumeField<Vec3>;
using SurfScalar = fvcc::SurfaceField<Scalar>;


TEST_CASE("timeIntegration: ddtPhiCorr on single-cell mesh", "[timeIntegration][ddtPhiCorr]")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("ddtPhiCorr basic identities on " + execName)
    {
        // --- 1) Database + single-cell mesh
        NeoN::Database db;
        auto mesh = NeoN::createSingleCellMesh(exec);

        fvcc::VectorCollection& fieldCollection =
            fvcc::VectorCollection::instance(db, "fieldCollection");

        // --- 2) Volume velocity field U (with oldTime support)
        auto& u = fieldCollection.registerVector<VolVector>(CreateVolumeVector<Vec3> {
            .name = "U", .mesh = mesh, .value = Vec3 {0.0, 0.0, 0.0}, .timeIndex = 1
        });

        // old-time U^0 := (2, 1, -0.5)
        auto& u0 = fvcc::oldTime(u);
        NeoN::fill(u0.internalVector(), Vec3 {2.0, 1.0, -0.5});

        // --- 3) Interpolate U0 to faces (reference)
        fvcc::SurfaceInterpolation<Vec3> interp(
            exec,
            mesh,
            NeoN::TokenList({std::string("linear")}) // TODO: read surfaceInterpolation from dict
        );
        auto uf0 = interp.interpolate(u0);

        auto [uf0BH, sfH] =
            copyToHosts(uf0.boundaryData().value(), mesh.boundaryMesh().faceNormals());
        auto [uf0BV, sfV] = views(uf0BH, sfH);

        // --- 4) Surface scalar field phi (with oldTime support)
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Scalar>>(mesh);

        auto& flux = fieldCollection.registerVector<SurfScalar>(CreateSurfaceVector<Scalar> {
            .name = "flux", .mesh = mesh, .bcs = &surfaceBCs, .value = 0.0, .timeIndex = 1
        });

        auto& flux0 = fvcc::oldTime(flux);

        // --- 5) DdtOperator + scheme selection
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(U)", std::string("BDF1"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, u);
        ddtOp.read(fvSchemes);

        auto scheme = ddtOp.scheme();

        const Scalar dt = 0.2;
        const Scalar invDt = 1.0 / dt;

        // For single-cell mesh all faces are boundary faces; the test uses nBoundaryFaces
        const auto nBndFaces = toLocalIdx(mesh.nBoundaryFaces());

        // Helper: expected values for boundary faces (host-side)
        auto expectedFrom = [&](const SurfScalar& flux0Field)
        {
            std::vector<Scalar> e(mesh.nBoundaryFaces(), 0.0);
            auto flux0FieldH = flux0Field.boundaryData().value().copyToHost();
            auto flux0FieldV = flux0FieldH.view();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                const Scalar d = (sfV[i] & uf0BV[i]);
                const auto tfluxCorr = (flux0FieldV[i] - d);
                const auto ratio = NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + 1.0e-30);
                const auto coeff = 1.0 - Kokkos::min(ratio, Scalar(1));

                e[static_cast<sizeType>(i)] = coeff * invDt * tfluxCorr;
            }
            return e;
        };

        // ─────────────────────────────────────────────
        // Case A: flux0 = sf·uf0  ⇒ correction ≈ 0
        // ─────────────────────────────────────────────
        {
            auto [flux0BV2, sfV2, uf0BV2] = views(
                flux0.boundaryData().value(),
                mesh.boundaryMesh().faceNormals(),
                uf0.boundaryData().value()
            );

            NeoN::parallelFor(
                exec,
                {0, nBndFaces},
                NEON_LAMBDA(const NeoN::localIdx i) { flux0BV2[i] = (sfV2[i] & uf0BV2[i]); }
            );

            flux.boundaryData().value() = flux0.boundaryData().value();

            auto fluxCorr = fvcc::ddtFluxCorr(u, flux, dt, scheme);
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                REQUIRE(corrBH.view()[i] == Catch::Approx(0.0).margin(1e-12));
            }
        }

        // ─────────────────────────────────────────────
        // Case B: flux0 = 0  ⇒ correction = -(sf·uf0)/dt
        // ─────────────────────────────────────────────
        {
            fill(flux0.boundaryData().value(), Scalar {0.0});
            fill(flux.boundaryData().value(), Scalar {0.0});

            auto expected = expectedFrom(flux0);
            auto fluxCorr = fvcc::ddtFluxCorr(u, flux, dt, scheme);
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                REQUIRE(
                    corrBH.view()[i]
                    == Catch::Approx(expected[static_cast<sizeType>(i)]).margin(1e-12)
                );
            }
        }

        // --- 6) Volume density field rho (with oldTime support), for the density-weighted
        // overload. Uniform rho0 so rhoUf0 = rho0 * uf0 is just a scaled copy of uf0.
        auto& rho = fieldCollection.registerVector<VolScalar>(CreateVolumeVector<Scalar> {
            .name = "rho", .mesh = mesh, .value = Scalar {0.0}, .timeIndex = 1
        });
        auto& rho0 = fvcc::oldTime(rho);
        NeoN::fill(rho0.internalVector(), Scalar {1.5});

        fvcc::SurfaceInterpolation<Scalar> interpScalar(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        auto rhoUf0 = interpScalar.interpolate(rho0) * uf0;

        auto rhoUf0BH = rhoUf0.boundaryData().value().copyToHost();
        auto rhoUf0BV = rhoUf0BH.view();

        auto expectedRhoFrom = [&](const SurfScalar& flux0Field)
        {
            std::vector<Scalar> e(static_cast<sizeType>(nBndFaces), 0.0);
            auto flux0FieldH = flux0Field.boundaryData().value().copyToHost();
            auto flux0FieldV = flux0FieldH.view();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                const Scalar d = (sfV[i] & rhoUf0BV[i]);
                const auto tfluxCorr = (flux0FieldV[i] - d);
                const auto ratio = NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + 1.0e-30);
                const auto coeff = 1.0 - Kokkos::min(ratio, Scalar(1));

                e[static_cast<sizeType>(i)] = coeff * invDt * tfluxCorr;
            }
            return e;
        };

        // ─────────────────────────────────────────────
        // Case C: flux0 = sf·(rho0*uf0)  ⇒ correction ≈ 0
        // ─────────────────────────────────────────────
        {
            auto [flux0BV2, sfV2, rhoUf0BV2] = views(
                flux0.boundaryData().value(),
                mesh.boundaryMesh().faceNormals(),
                rhoUf0.boundaryData().value()
            );

            NeoN::parallelFor(
                exec,
                {sizeType(0), static_cast<sizeType>(nBndFaces)},
                NEON_LAMBDA(const NeoN::localIdx i) { flux0BV2[i] = (sfV2[i] & rhoUf0BV2[i]); }
            );

            flux.boundaryData().value() = flux0.boundaryData().value();

            auto fluxCorr = fvcc::ddtFluxCorr(rho, u, flux, dt, scheme);
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                REQUIRE(corrBH.view()[i] == Catch::Approx(0.0).margin(1e-12));
            }
        }

        // ─────────────────────────────────────────────
        // Case D: flux0 = 0  ⇒ correction = -(sf·rho0*uf0)/dt
        // ─────────────────────────────────────────────
        {
            fill(flux0.boundaryData().value(), Scalar {0.0});
            fill(flux.boundaryData().value(), Scalar {0.0});

            auto expected = expectedRhoFrom(flux0);
            auto fluxCorr = fvcc::ddtFluxCorr(rho, u, flux, dt, scheme);
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                REQUIRE(
                    corrBH.view()[i]
                    == Catch::Approx(expected[static_cast<sizeType>(i)]).margin(1e-12)
                );
            }
        }
    }

    SECTION(
        "density-weighted ddtFluxCorr with non-trivial old-time values, BDF1 and BDF2 on "
        + execName
    )
    {
        // The identities above leave the boundary values of U^0 and rho^0 at zero (the
        // helper creates the fields without boundary conditions), so they cannot tell a
        // correctly weighted correction from an unweighted one, and they only run BDF1.
        // Here every old-time field carries distinct, non-zero boundary values and the
        // expected correction is computed independently on the host from the formulas:
        //   BDF1: limiter(phi^0, corr^0) * (1/dt) * corr^0
        //   BDF2: limiter(phi^0, corr^0) * (2/dt) * corr^0
        //       + limiter(phi^00, corr^00) * (-1/(2 dt)) * corr^00
        // with corr^k = phi^k - Sf & (rho_f^k * uf^k).
        NeoN::Database db;
        auto mesh = NeoN::createSingleCellMesh(exec);
        fvcc::VectorCollection& fieldCollection =
            fvcc::VectorCollection::instance(db, "fieldCollection");

        auto& u = fieldCollection.registerVector<VolVector>(CreateVolumeVector<Vec3> {
            .name = "U", .mesh = mesh, .value = Vec3 {0.0, 0.0, 0.0}, .timeIndex = 1
        });
        auto& u0 = fvcc::oldTime(u);
        auto& u00 = fvcc::oldTime(u0); // two old-time levels: enables the BDF2 branch

        auto& rho = fieldCollection.registerVector<VolScalar>(CreateVolumeVector<Scalar> {
            .name = "rho", .mesh = mesh, .value = Scalar {0.0}, .timeIndex = 1
        });
        auto& rho0 = fvcc::oldTime(rho);
        auto& rho00 = fvcc::oldTime(rho0);

        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Scalar>>(mesh);
        auto& flux = fieldCollection.registerVector<SurfScalar>(CreateSurfaceVector<Scalar> {
            .name = "flux", .mesh = mesh, .bcs = &surfaceBCs, .value = 0.0, .timeIndex = 1
        });
        auto& flux0 = fvcc::oldTime(flux);
        auto& flux00 = fvcc::oldTime(flux0);

        // rho^0 != rho^00 and U^0 != U^00, so using the wrong level changes the result
        const Vec3 uOld {2.0, 1.0, -0.5};
        const Vec3 uOldOld {-1.0, 3.0, 0.5};
        const Scalar rhoOld = 1.5;
        const Scalar rhoOldOld = 0.5;
        for (auto [field, value] : {std::pair<VolVector*, Vec3> {&u0, uOld}, {&u00, uOldOld}})
        {
            NeoN::fill(field->internalVector(), value);
            NeoN::fill(field->boundaryData().value(), value);
        }
        for (auto [field, value] :
             {std::pair<VolScalar*, Scalar> {&rho0, rhoOld}, {&rho00, rhoOldOld}})
        {
            NeoN::fill(field->internalVector(), value);
            NeoN::fill(field->boundaryData().value(), value);
        }

        // rho_f^k * uf^k, as the overload builds it
        fvcc::SurfaceInterpolation<Vec3> interpU(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        fvcc::SurfaceInterpolation<Scalar> interpRho(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        auto rhoUf0 = interpRho.interpolate(rho0) * interpU.interpolate(u0);
        auto rhoUf00 = interpRho.interpolate(rho00) * interpU.interpolate(u00);

        const auto nBndFaces = toLocalIdx(mesh.nBoundaryFaces());
        const Scalar dt = 0.2;

        // phi^0 = 1.1 * Sf & rhoUf^0 and phi^00 = 0.8 * Sf & rhoUf^00: a consistent flux
        // perturbed by +10% / -20%, so every limiter lies strictly between 0 and 1
        {
            auto [flux0BV, flux00BV, sfV, rhoUf0BV, rhoUf00BV] = views(
                flux0.boundaryData().value(),
                flux00.boundaryData().value(),
                mesh.boundaryMesh().faceNormals(),
                rhoUf0.boundaryData().value(),
                rhoUf00.boundaryData().value()
            );
            NeoN::parallelFor(
                exec,
                {0, nBndFaces},
                NEON_LAMBDA(const NeoN::localIdx i) {
                    flux0BV[i] = 1.1 * (sfV[i] & rhoUf0BV[i]);
                    flux00BV[i] = 0.8 * (sfV[i] & rhoUf00BV[i]);
                }
            );
        }
        flux.boundaryData().value() = flux0.boundaryData().value();

        auto [sfH, rhoUf0H, rhoUf00H] = copyToHosts(
            mesh.boundaryMesh().faceNormals(),
            rhoUf0.boundaryData().value(),
            rhoUf00.boundaryData().value()
        );
        auto flux0H = flux0.boundaryData().value().copyToHost();
        auto flux00H = flux00.boundaryData().value().copyToHost();

        auto limiter = [](Scalar fluxValue, Scalar corr)
        {
            const Scalar ratio = NeoN::mag(corr) / (NeoN::mag(fluxValue) + 1.0e-30);
            return 1.0 - Kokkos::min(ratio, Scalar(1));
        };

        std::vector<Scalar> expectedBDF1(static_cast<sizeType>(nBndFaces), 0.0);
        std::vector<Scalar> expectedBDF2(static_cast<sizeType>(nBndFaces), 0.0);
        for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
        {
            const Scalar corr0 = flux0H.view()[i] - (sfH.view()[i] & rhoUf0H.view()[i]);
            const Scalar corr00 = flux00H.view()[i] - (sfH.view()[i] & rhoUf00H.view()[i]);
            const Scalar term0 = limiter(flux0H.view()[i], corr0) * corr0;
            const Scalar term00 = limiter(flux00H.view()[i], corr00) * corr00;

            expectedBDF1[static_cast<sizeType>(i)] = term0 / dt;
            expectedBDF2[static_cast<sizeType>(i)] = 2.0 * term0 / dt - 0.5 * term00 / dt;

            // the expectation must be non-trivial, and BDF2 must differ from BDF1
            REQUIRE(NeoN::mag(expectedBDF1[static_cast<sizeType>(i)]) > 1e-3);
            REQUIRE(
                NeoN::mag(
                    expectedBDF2[static_cast<sizeType>(i)] - expectedBDF1[static_cast<sizeType>(i)]
                )
                > 1e-3
            );
        }

        for (const std::string schemeName : {"BDF1", "BDF2"})
        {
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(U)", schemeName);
            fvSchemes.insert("ddtSchemes", ddtSchemes);
            fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, u);
            ddtOp.read(fvSchemes);

            const auto& expected = schemeName == "BDF1" ? expectedBDF1 : expectedBDF2;
            auto fluxCorr = fvcc::ddtFluxCorr(rho, u, flux, dt, ddtOp.scheme());
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                INFO(schemeName << ", boundary face " << i);
                REQUIRE(
                    corrBH.view()[i]
                    == Catch::Approx(expected[static_cast<sizeType>(i)]).margin(1e-12)
                );
            }
        }
    }
}
