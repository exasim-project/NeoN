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

        auto [uf0H, sfH] = copyToHosts(uf0.internalVector(), mesh.boundaryMesh().sf());
        auto [uf0V, sfV] = views(uf0H, sfH);

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

        const auto& scheme = ddtOp.scheme();

        const Scalar dt = 0.2;
        const Scalar invDt = 1.0 / dt;

        // Helper: expected values (host-side)
        auto expectedFrom = [&](const SurfScalar& flux0Field)
        {
            std::vector<Scalar> e(mesh.nFaces(), 0.0);
            auto flux0FieldH = flux0Field.internalVector().copyToHost();
            auto flux0FieldV = flux0FieldH.view();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
            {
                const Scalar d = (sfV[i] & uf0V[i]);
                const auto tfluxCorr = (flux0FieldV[i] - d);
                const auto ratio =
                    NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + Scalar(1e-30));
                const auto coeff = Scalar(1.0) - Kokkos::min(ratio, Scalar(1));

                e[i] = coeff * invDt * tfluxCorr;
            }
            return e;
        };

        // ─────────────────────────────────────────────
        // Case A: flux0 = sf·uf0  ⇒ correction ≈ 0
        // ─────────────────────────────────────────────
        {
            auto [flux0V, sfV, uf0V] =
                views(flux0.internalVector(), mesh.boundaryMesh().sf(), uf0.internalVector());

            NeoN::parallelFor(
                exec,
                {size_t(0), mesh.nFaces()},
                KOKKOS_LAMBDA(const NeoN::localIdx i) { flux0V[i] = (sfV[i] & uf0V[i]); }
            );

            flux.internalVector() = flux0.internalVector();

            auto fluxCorr = scheme.ddtFluxCorr(u, flux, dt);
            auto corrH = fluxCorr.internalVector().copyToHost();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
                REQUIRE(corrH.view()[i] == Approx(0.0).margin(1e-12));
        }

        // ─────────────────────────────────────────────
        // Case B: flux0 = 0  ⇒ correction = -(sf·uf0)/dt
        // ─────────────────────────────────────────────
        {
            flux0.internalVector() = Scalar {0.0};
            flux.internalVector() = Scalar {0.0};

            auto expected = expectedFrom(flux0);
            auto fluxCorr = scheme.ddtFluxCorr(u, flux, dt);
            auto corrH = fluxCorr.internalVector().copyToHost();

            for (size_t i = 0; i < mesh.nFaces(); ++i)
                REQUIRE(corrH.view()[i] == Approx(expected[i]).margin(1e-12));
        }
    }
}
