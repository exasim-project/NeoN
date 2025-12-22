// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "../dsl/common.hpp"
#include "NeoN/NeoN.hpp"

using Catch::Approx;
namespace fvcc = NeoN::finiteVolume::cellCentred;

using Scalar = NeoN::scalar;
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
        auto& U = fieldCollection.registerVector<VolVector>(CreateVolumeVector<Vec3> {
            .name = "U", .mesh = mesh, .value = Vec3 {0.0, 0.0, 0.0}, .timeIndex = 1
        });

        // old-time U^0 := (2, 1, -0.5)
        auto& U0 = fvcc::oldTime(U);
        NeoN::fill(U0.internalVector(), Vec3 {2.0, 1.0, -0.5});

        // --- 3) Interpolate U0 to faces (reference)
        fvcc::SurfaceInterpolation<Vec3> interp(
            exec, mesh, NeoN::TokenList({std::string("linear")})
        );
        auto Uf0f = interp.interpolate(U0);

        auto Uf0fHost = Uf0f.internalVector().copyToHost();
        auto UfView = Uf0fHost.view();

        auto SfHost = mesh.boundaryMesh().sf().copyToHost();
        auto Sf = SfHost.view();

        // --- 4) Surface scalar field phi (with oldTime support)
        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Scalar>>(mesh);

        auto& phi = fieldCollection.registerVector<SurfScalar>(CreateSurfaceVector<Scalar> {
            .name = "phi", .mesh = mesh, .bcs = &surfaceBCs, .value = 0.0, .timeIndex = 1
        });

        auto& phi0 = fvcc::oldTime(phi);

        // --- 5) DdtOperator + scheme selection
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        NeoN::Dictionary timeIntegrationDict;
        timeIntegrationDict.insert("type", std::string("backwardEuler"));
        ddtSchemes.insert("ddt(U)", std::string("BDF1"));
        fvSchemes.insert("timeIntegration", timeIntegrationDict);
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        fvcc::DdtOperator<Vec3> ddtOp(NeoN::dsl::Operator::Type::Implicit, U);
        ddtOp.read(fvSchemes);

        const auto& scheme = ddtOp.scheme();

        const Scalar dt = 0.2;
        const Scalar invDt = 1.0 / dt;

        // Helper: expected values (host-side)
        auto expected_from = [&](const SurfScalar& phi0Field)
        {
            std::vector<Scalar> e(mesh.nFaces(), 0.0);
            auto ph0Host = phi0Field.internalVector().copyToHost();
            auto ph0v = ph0Host.view();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
            {
                const Scalar d = (Sf[i] & UfView[i]);
                const auto tphiCorr = (ph0v[i] - d);
                const auto ratio = NeoN::mag(tphiCorr) / (NeoN::mag(ph0v[i]) + Scalar(1e-30));
                const auto coeff = Scalar(1.0) - Kokkos::min(ratio, Scalar(1));

                e[i] = coeff * invDt * tphiCorr;
            }
            return e;
        };

        // ─────────────────────────────────────────────
        // Case A: phi0 = Sf·Uf0f  ⇒ correction ≈ 0
        // ─────────────────────────────────────────────
        {
            auto phi0_v = phi0.internalVector().view();
            auto Sf_v = mesh.boundaryMesh().sf().view();
            auto Uf_v = Uf0f.internalVector().view();

            NeoN::parallelFor(
                exec,
                {size_t(0), mesh.nFaces()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) { phi0_v[i] = (Sf_v[i] & Uf_v[i]); }
            );

            phi.internalVector() = phi0.internalVector();

            auto phiCorr = scheme.ddtPhiCorr(U, phi, dt);
            auto corrHost = phiCorr.internalVector().copyToHost();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
                REQUIRE(corrHost.view()[i] == Approx(0.0).margin(1e-12));
        }

        // ─────────────────────────────────────────────
        // Case B: phi0 = 0  ⇒ correction = -(Sf·Uf0f)/dt
        // ─────────────────────────────────────────────
        {
            phi0.internalVector() = Scalar {0.0};
            phi.internalVector() = Scalar {0.0};

            auto expected = expected_from(phi0);
            auto phiCorr = scheme.ddtPhiCorr(U, phi, dt);
            auto corrHost = phiCorr.internalVector().copyToHost();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
                REQUIRE(corrHost.view()[i] == Approx(expected[i]).margin(1e-12));
        }
    }
}
