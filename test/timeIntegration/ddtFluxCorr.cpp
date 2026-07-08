// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "../dsl/common.hpp"
#include "NeoN/NeoN.hpp"

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
        const auto nBndFaces = mesh.nBoundaryFaces();

        // Helper: expected values for boundary faces (host-side)
        auto expectedFrom = [&](const SurfScalar& flux0Field)
        {
            std::vector<Scalar> e(static_cast<size_t>(nBndFaces), 0.0);
            auto flux0FieldH = flux0Field.boundaryData().value().copyToHost();
            auto flux0FieldV = flux0FieldH.view();

            for (NeoN::localIdx i = 0; i < nBndFaces; ++i)
            {
                const Scalar d = (sfV[i] & uf0BV[i]);
                const auto tfluxCorr = (flux0FieldV[i] - d);
                const auto ratio = NeoN::mag(tfluxCorr) / (NeoN::mag(flux0FieldV[i]) + 1.0e-30);
                const auto coeff = 1.0 - Kokkos::min(ratio, Scalar(1));

                e[static_cast<size_t>(i)] = coeff * invDt * tfluxCorr;
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
                {size_t(0), static_cast<size_t>(nBndFaces)},
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
                    == Catch::Approx(expected[static_cast<size_t>(i)]).margin(1e-12)
                );
            }
        }
    }
}
