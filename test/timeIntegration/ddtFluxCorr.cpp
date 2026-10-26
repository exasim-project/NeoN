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
                const auto ratio = NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + 1.0e-15);
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
                const auto ratio = NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + 1.0e-15);
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
        //   BDF2: limiter(phi^0, corr^0) * (1/dt) * (2 corr^0 - corr^00 / 2)
        // with corr^k = phi^k - Sf & (rho_f^k * uf^k): one coupling coefficient, from corr^0
        // alone, as in OpenFOAM's backwardDdtScheme::fvcDdtPhiCorr. The density-weighted overload
        // is zero on the boundary faces, see below.
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
            const Scalar ratio = NeoN::mag(corr) / (NeoN::mag(fluxValue) + 1.0e-15);
            return 1.0 - Kokkos::min(ratio, Scalar(1));
        };

        std::vector<Scalar> expectedBDF1(static_cast<sizeType>(nBndFaces), 0.0);
        std::vector<Scalar> expectedBDF2(static_cast<sizeType>(nBndFaces), 0.0);
        for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
        {
            const Scalar corr0 = flux0H.view()[i] - (sfH.view()[i] & rhoUf0H.view()[i]);
            const Scalar corr00 = flux00H.view()[i] - (sfH.view()[i] & rhoUf00H.view()[i]);
            // a single coupling coefficient, from the single-level correction corr^0
            const Scalar coeff = limiter(flux0H.view()[i], corr0);

            expectedBDF1[static_cast<sizeType>(i)] = coeff * corr0 / dt;
            expectedBDF2[static_cast<sizeType>(i)] = coeff * (2.0 * corr0 - 0.5 * corr00) / dt;

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

            // OpenFOAM evaluates the coupling coefficient on the rho*U product, whose
            // 'calculated' patches fix their value: the correction is zero on every physical
            // boundary face. The unzeroed value (expected above) is non-trivial, so this checks
            // the zeroing; the weighting itself is checked on internal faces further below.
            auto fluxCorr = fvcc::ddtFluxCorr(rho, u, flux, dt, ddtOp.scheme());
            auto corrBH = fluxCorr.boundaryData().value().copyToHost();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                INFO(schemeName << ", boundary face " << i);
                REQUIRE(corrBH.view()[i] == Catch::Approx(0.0).margin(1e-14));
            }
        }
    }
}

TEST_CASE(
    "timeIntegration: ddtFluxCorr on a 1D mesh with internal faces", "[timeIntegration][ddtPhiCorr]"
)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("incompressible and density-weighted, BDF1 and BDF2 on " + execName)
    {
        // The expected values are computed on the host from OpenFOAM's definition
        // (ddtScheme::fvcDdtPhiCoeff, EulerDdtScheme/backwardDdtScheme::fvcDdtPhiCorr):
        //   corr^k = phi^k - Sf & uf^k
        //   coeff  = 1 - min(|corr^0| / (|phi^0| + SMALL), 1)
        //   BDF1:    coeff * corr^0 / dt
        //   BDF2:    coeff * (2 corr^0 - corr^00 / 2) / dt
        // with uf^k = interpolate(U^k) and, for the density-weighted overload,
        // uf^k = interpolate(rho^k * U^k): the product is formed in the cells and then
        // interpolated, which is not interpolate(rho^k) * interpolate(U^k) as soon as rho and U
        // vary between neighbouring cells. Single-cell meshes cannot see that difference, since
        // they only have boundary faces.
        NeoN::Database db;
        const NeoN::localIdx nCells = 5;
        auto mesh = NeoN::create1DUniformMesh(exec, nCells);
        fvcc::VectorCollection& fieldCollection =
            fvcc::VectorCollection::instance(db, "fieldCollection");

        auto& u = fieldCollection.registerVector<VolVector>(CreateVolumeVector<Vec3> {
            .name = "U", .mesh = mesh, .value = Vec3 {0.0, 0.0, 0.0}, .timeIndex = 1
        });
        auto& u0 = fvcc::oldTime(u);
        auto& u00 = fvcc::oldTime(u0);

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

        const auto nInt = static_cast<std::size_t>(mesh.nInternalFaces());
        const auto nBnd = static_cast<std::size_t>(mesh.nBoundaryFaces());
        const auto nC = static_cast<std::size_t>(nCells);
        const Scalar dt = 0.2;
        const Scalar small = 1.0e-15;

        // cell and boundary values that vary non-linearly from cell to cell, with different
        // values on the two old-time levels
        std::vector<Scalar> rho0H(nC), rho00H(nC), rho0B(nBnd), rho00B(nBnd);
        std::vector<Vec3> u0H(nC), u00H(nC), u0B(nBnd), u00B(nBnd);
        for (std::size_t i = 0; i < nC; ++i)
        {
            const auto x = static_cast<Scalar>(i);
            rho0H[i] = 1.0 + 0.15 * x;
            rho00H[i] = 0.7 + 0.2 * x * x;
            u0H[i] = Vec3 {1.0 + 0.4 * x, 0.3 * x - 0.5, 0.1 * x * x};
            u00H[i] = Vec3 {0.5 + 0.25 * x * x, 1.0 - 0.2 * x, 0.2 * x};
        }
        for (std::size_t b = 0; b < nBnd; ++b)
        {
            const auto x = static_cast<Scalar>(b);
            rho0B[b] = 1.2 + 0.3 * x;
            rho00B[b] = 0.9 + 0.4 * x;
            u0B[b] = Vec3 {1.5 + 0.7 * x, -0.4 + 0.2 * x, 0.3 * x};
            u00B[b] = Vec3 {0.8 + 0.3 * x, 0.5 - 0.6 * x, 0.1 + 0.2 * x};
        }
        rho0.internalVector() = NeoN::Vector<Scalar>(exec, rho0H);
        rho00.internalVector() = NeoN::Vector<Scalar>(exec, rho00H);
        rho0.boundaryData().value() = NeoN::Vector<Scalar>(exec, rho0B);
        rho00.boundaryData().value() = NeoN::Vector<Scalar>(exec, rho00B);
        u0.internalVector() = NeoN::Vector<Vec3>(exec, u0H);
        u00.internalVector() = NeoN::Vector<Vec3>(exec, u00H);
        u0.boundaryData().value() = NeoN::Vector<Vec3>(exec, u0B);
        u00.boundaryData().value() = NeoN::Vector<Vec3>(exec, u00B);

        // host copies of the geometry. Internal faces come first in the face lists; the mesh is
        // uniform, so linear interpolation weights every neighbour with 1/2.
        auto sfInternalH = mesh.faceNormals().copyToHost();
        auto sfBoundaryH = mesh.boundaryMesh().faceNormals().copyToHost();
        auto ownerH = mesh.faceOwners().copyToHost();
        auto neighbourH = mesh.faceNeighbors().copyToHost();
        auto sfI = sfInternalH.view();
        auto sfB = sfBoundaryH.view();
        auto own = ownerH.view();
        auto nei = neighbourH.view();

        // face values of U^k (incompressible) or rho^k * U^k (density-weighted)
        auto faceInternal = [&](bool weighted,
                                const std::vector<Scalar>& r,
                                const std::vector<Vec3>& v,
                                std::size_t f)
        {
            const auto o = static_cast<std::size_t>(own[f]);
            const auto n = static_cast<std::size_t>(nei[f]);
            return weighted ? 0.5 * (r[o] * v[o]) + 0.5 * (r[n] * v[n]) : 0.5 * v[o] + 0.5 * v[n];
        };
        auto faceBoundary = [&](bool weighted,
                                const std::vector<Scalar>& r,
                                const std::vector<Vec3>& v,
                                std::size_t b) { return weighted ? r[b] * v[b] : v[b]; };

        // fluxes that differ from Sf & uf by a different factor on every face: the coupling
        // coefficient takes values strictly between 0 and 1 and also gets clamped to 0
        const std::vector<Scalar> factor0 {1.1, 1.6, 0.2, 1.3, 0.9, 1.2, 1.05};
        const std::vector<Scalar> factor00 {0.8, 1.4, 1.7, 0.6, 1.2, 0.95, 1.5};

        const std::string schemeNames[] = {"BDF1", "BDF2"};
        for (const bool weighted : {false, true})
        {
            // phi^0 and phi^00 for this variant
            std::vector<Scalar> phi0I(nInt), phi00I(nInt), phi0B(nBnd), phi00B(nBnd);
            for (std::size_t f = 0; f < nInt; ++f)
            {
                phi0I[f] = factor0[f % 7] * (sfI[f] & faceInternal(weighted, rho0H, u0H, f));
                phi00I[f] = factor00[f % 7] * (sfI[f] & faceInternal(weighted, rho00H, u00H, f));
            }
            for (std::size_t b = 0; b < nBnd; ++b)
            {
                phi0B[b] = factor0[(b + 3) % 7] * (sfB[b] & faceBoundary(weighted, rho0B, u0B, b));
                phi00B[b] =
                    factor00[(b + 3) % 7] * (sfB[b] & faceBoundary(weighted, rho00B, u00B, b));
            }
            flux0.internalVector() = NeoN::Vector<Scalar>(exec, phi0I);
            flux00.internalVector() = NeoN::Vector<Scalar>(exec, phi00I);
            flux0.boundaryData().value() = NeoN::Vector<Scalar>(exec, phi0B);
            flux00.boundaryData().value() = NeoN::Vector<Scalar>(exec, phi00B);

            auto coupling = [&](Scalar phi, Scalar corr)
            { return 1.0 - std::min(std::abs(corr) / (std::abs(phi) + small), 1.0); };

            for (const auto& schemeName : schemeNames)
            {
                NeoN::Dictionary fvSchemes;
                NeoN::Dictionary ddtSchemes;
                ddtSchemes.insert("ddt(U)", schemeName);
                fvSchemes.insert("ddtSchemes", ddtSchemes);
                fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, u);
                ddtOp.read(fvSchemes);
                const bool bdf2 = schemeName == "BDF2";

                auto fluxCorr = weighted ? fvcc::ddtFluxCorr(rho, u, flux, dt, ddtOp.scheme())
                                         : fvcc::ddtFluxCorr(u, flux, dt, ddtOp.scheme());
                auto internalH = fluxCorr.internalVector().copyToHost();
                auto boundaryH = fluxCorr.boundaryData().value().copyToHost();

                std::size_t nStrictlyBetween = 0;
                std::size_t nClamped = 0;
                for (std::size_t f = 0; f < nInt; ++f)
                {
                    const Scalar corr0 =
                        phi0I[f] - (sfI[f] & faceInternal(weighted, rho0H, u0H, f));
                    const Scalar corr00 =
                        phi00I[f] - (sfI[f] & faceInternal(weighted, rho00H, u00H, f));
                    const Scalar coeff = coupling(phi0I[f], corr0);
                    const Scalar expected =
                        coeff * (bdf2 ? 2.0 * corr0 - 0.5 * corr00 : corr0) / dt;
                    nStrictlyBetween += (coeff > 0.05 && coeff < 0.95) ? 1 : 0;
                    nClamped += (coeff == 0.0) ? 1 : 0;

                    INFO(
                        (weighted ? "density-weighted " : "incompressible ")
                        << schemeName << ", internal face " << f
                    );
                    REQUIRE(internalH.view()[f] == Catch::Approx(expected).margin(1e-12));
                }
                for (std::size_t b = 0; b < nBnd; ++b)
                {
                    const Scalar corr0 =
                        phi0B[b] - (sfB[b] & faceBoundary(weighted, rho0B, u0B, b));
                    const Scalar corr00 =
                        phi00B[b] - (sfB[b] & faceBoundary(weighted, rho00B, u00B, b));
                    const Scalar coeff = coupling(phi0B[b], corr0);
                    // density-weighted: zero on every physical boundary face (the rho*U product
                    // has 'calculated' patches, which fix their value in OpenFOAM)
                    const Scalar expected =
                        weighted ? 0.0 : coeff * (bdf2 ? 2.0 * corr0 - 0.5 * corr00 : corr0) / dt;

                    INFO(
                        (weighted ? "density-weighted " : "incompressible ")
                        << schemeName << ", boundary face " << b
                    );
                    REQUIRE(boundaryH.view()[b] == Catch::Approx(expected).margin(1e-12));
                }
                // the data must exercise the coupling coefficient, not just the identity cases
                REQUIRE(nStrictlyBetween >= 1);
                REQUIRE(nClamped >= 1);
            }

            if (weighted)
            {
                // the data must also tell interpolate(rho * U) from interpolate(rho) *
                // interpolate(U)
                Scalar maxDifference = 0.0;
                for (std::size_t f = 0; f < nInt; ++f)
                {
                    const auto o = static_cast<std::size_t>(own[f]);
                    const auto n = static_cast<std::size_t>(nei[f]);
                    const Vec3 productOfInterpolants =
                        (0.5 * rho0H[o] + 0.5 * rho0H[n]) * (0.5 * u0H[o] + 0.5 * u0H[n]);
                    const Vec3 interpolatedProduct = faceInternal(true, rho0H, u0H, f);
                    maxDifference = std::max(
                        maxDifference,
                        std::abs(sfI[f] & (interpolatedProduct - productOfInterpolants))
                    );
                }
                REQUIRE(maxDifference > 1e-3);
            }
        }
    }
}
