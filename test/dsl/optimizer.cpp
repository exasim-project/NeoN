// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "common.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

TEST_CASE("Optimizer")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = createSingleCellMesh(exec);
    auto sp = la::SparsityPattern {mesh};

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, 1, 2.0 * one<scalar>()), volBCs
    );

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);

    SECTION("Can optimize div + laplacian " + execName)
    {
        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::exp::div(phi, U);
        REQUIRE(expr.size() == 2);
        auto exprOpt = dsl::optimize(expr);
        REQUIRE(exprOpt.size() == 1);
    }

    SECTION("Can optimize div + laplacian " + execName)
    {
        auto input = NeoN::Dictionary {
            {
                "laplacianSchemes",
                NeoN::Dictionary {
                    {"laplacian(gamma,U)",
                     NeoN::TokenList(
                         {std::string("Gauss"), std::string("linear"), std::string("uncorrected")}
                     )}
                },
            },
            {"divSchemes",
             NeoN::Dictionary {
                 {"div(phi,U)", NeoN::TokenList({std::string("Gauss"), std::string("upwind")})}
             }}
        };

        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::exp::div(phi, U);
        auto exprOpt = dsl::optimize(expr);
        expr.read(input);
    }
}


}
