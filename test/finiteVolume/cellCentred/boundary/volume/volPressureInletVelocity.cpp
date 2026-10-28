// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("pressureInletVelocity_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 3 x 2 x 2 cells of 0.5 x 0.5 x 0.25. Patch 0 is x = 0: 4 faces of area 0.125, outward
    // normal (-1, 0, 0).
    auto mesh = NeoN::create3DUniformMesh(exec, 3, 2, 2, 1.5, 1.0, 0.5);
    const auto offsets = mesh.boundaryMesh().offset();
    const std::pair<NeoN::localIdx, NeoN::localIdx> patch0Range {offsets[0], offsets[1]};
    const NeoN::scalar area = 0.125;
    const NeoN::Vec3 untouched {7.0, 7.0, 7.0};

    // per-face flux and density of patch 0, different on every face
    const std::vector<NeoN::scalar> phiPatch {-0.02, -0.01, 0.015, 0.0};
    const std::vector<NeoN::scalar> rhoPatch {1.0, 1.2, 0.9, 1.1};

    auto makeVelocity = [&]()
    {
        std::vector<fvcc::VolumeBoundary<NeoN::Vec3>> bcs {};
        for (NeoN::localIdx patchi = 0; patchi < mesh.boundaryMesh().nBoundaries(); ++patchi)
        {
            NeoN::Dictionary dict;
            if (patchi == 0)
            {
                dict.insert("type", std::string("pressureInletVelocity"));
            }
            else
            {
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", untouched);
            }
            bcs.push_back(fvcc::VolumeBoundary<NeoN::Vec3>(mesh, dict, patchi));
        }
        fvcc::VolumeField<NeoN::Vec3> U(exec, "U", mesh, bcs);
        NeoN::fill(U.internalVector(), NeoN::Vec3 {1.0, 2.0, 3.0});
        NeoN::fill(U.boundaryData().value(), untouched);
        return U;
    };

    auto setPatch0 = [&](auto& field, const std::vector<NeoN::scalar>& values)
    {
        std::vector<NeoN::scalar> all(static_cast<std::size_t>(mesh.nBoundaryFaces()), 0.0);
        for (std::size_t k = 0; k < values.size(); ++k)
            all[static_cast<std::size_t>(patch0Range.first) + k] = values[k];
        field.boundaryData().value() = NeoN::Vector<NeoN::scalar>(exec, all);
    };

    auto phi = fvcc::SurfaceField<NeoN::scalar>(
        exec, "phi", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
    );
    NeoN::fill(phi.internalVector(), NeoN::scalar(0));
    setPatch0(phi, phiPatch);
    auto rho = fvcc::VolumeField<NeoN::scalar>(
        exec, "rho", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh)
    );
    NeoN::fill(rho.internalVector(), NeoN::scalar(1));
    setPatch0(rho, rhoPatch);

    SECTION("a mass flux gives n * phi / (rho * |S|) " + execName)
    {
        auto U = makeVelocity();
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("rho", rho);
        U.correctBoundaryConditions(ctx);

        auto [valueH, refValueH, valueFractionH] = copyToHosts(
            U.boundaryData().value(), U.boundaryData().refValue(), U.boundaryData().valueFraction()
        );
        for (NeoN::localIdx k = 0; k < 4; ++k)
        {
            const auto i = patch0Range.first + k;
            const NeoN::scalar un = phiPatch[static_cast<std::size_t>(k)]
                                  / (rhoPatch[static_cast<std::size_t>(k)] * area);
            // outward normal (-1, 0, 0): an inflow (phi < 0) points into the domain, +x
            REQUIRE(valueH.view()[i][0] == Catch::Approx(-un).margin(1e-12));
            REQUIRE(valueH.view()[i][1] == Catch::Approx(0.0).margin(1e-12));
            REQUIRE(valueH.view()[i][2] == Catch::Approx(0.0).margin(1e-12));
            REQUIRE(refValueH.view()[i][0] == valueH.view()[i][0]);
            REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
        }
        // the other patches are not touched
        for (auto i = offsets[1]; i < offsets.back(); ++i)
            REQUIRE(valueH.view()[i][0] == untouched[0]);
    }

    SECTION("without a density the flux is volumetric " + execName)
    {
        auto U = makeVelocity();
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        U.correctBoundaryConditions(ctx);

        auto valueH = U.boundaryData().value().copyToHost();
        for (NeoN::localIdx k = 0; k < 4; ++k)
        {
            const auto i = patch0Range.first + k;
            REQUIRE(
                valueH.view()[i][0]
                == Catch::Approx(-phiPatch[static_cast<std::size_t>(k)] / area).margin(1e-12)
            );
        }
    }

    SECTION("without a flux the patch is at rest " + execName)
    {
        auto U = makeVelocity();
        U.correctBoundaryConditions();
        auto valueH = U.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            for (int c = 0; c < 3; ++c)
                REQUIRE(valueH.view()[i][c] == 0.0);

        // and so does a context that lacks the flux
        fvcc::BoundaryContext ctx;
        U.correctBoundaryConditions(ctx);
        auto again = U.boundaryData().value().copyToHost();
        for (auto i = patch0Range.first; i < patch0Range.second; ++i)
            REQUIRE(again.view()[i][0] == 0.0);
    }

    SECTION("attributes follow OpenFOAM " + execName)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("pressureInletVelocity"));
        auto bc =
            fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create("pressureInletVelocity", mesh, dict, 0);
        // OpenFOAM's pressureInletVelocity is assignable(): constrainHbyA leaves HbyA alone
        REQUIRE(bc->attributes().assignable == true);
        REQUIRE(bc->attributes().fixesValue == true);
    }

    SECTION("the flux and density names can be set " + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::Vec3>> bcs {};
        for (NeoN::localIdx patchi = 0; patchi < mesh.boundaryMesh().nBoundaries(); ++patchi)
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("pressureInletVelocity"));
            dict.insert("phi", std::string("phiOther"));
            dict.insert("rho", std::string("rhoOther"));
            bcs.push_back(fvcc::VolumeBoundary<NeoN::Vec3>(mesh, dict, patchi));
        }
        fvcc::VolumeField<NeoN::Vec3> U(exec, "U", mesh, bcs);
        fvcc::BoundaryContext ctx;
        ctx.insert("phiOther", phi);
        ctx.insert("rhoOther", rho);
        U.correctBoundaryConditions(ctx);
        auto valueH = U.boundaryData().value().copyToHost();
        REQUIRE(
            valueH.view()[patch0Range.first][0]
            == Catch::Approx(-phiPatch[0] / (rhoPatch[0] * area)).margin(1e-12)
        );
    }
}
