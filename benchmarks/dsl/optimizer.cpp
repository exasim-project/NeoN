// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

namespace dsl = NeoN::dsl;


namespace NeoN
{

TEST_CASE("Assemble<DivLap>::unoptimized", "[bench]")
{
    float epsilon = 1e-32;
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);

    fill(phi.internalVector(), 1.0);
    fill(gamma.internalVector(), 2.0);

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

    DYNAMIC_SECTION("" << size)
    {

        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::imp::div(phi, U);
        expr.read(input);

        BENCHMARK(std::string(execName)) { return expr.assemble(mesh, 1.0, 1.0); };
    }
}

TEST_CASE("Assemble<DivLap>::fused", "[bench]")
{
    float epsilon = 1e-32;
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    auto U = finiteVolume::cellCentred::VolumeField<scalar>(
        exec, "U", mesh, Vector<scalar>(exec, nCells, 2.0 * one<scalar>()), volBCs
    );

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    auto phi = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "phi", mesh, surfaceBCs);
    auto gamma = finiteVolume::cellCentred::SurfaceField<scalar>(exec, "gamma", mesh, surfaceBCs);

    fill(phi.internalVector(), 1.0);
    fill(gamma.internalVector(), 2.0);

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

    DYNAMIC_SECTION("" << size)
    {

        auto expr = NeoN::dsl::imp::laplacian(gamma, U) - NeoN::dsl::imp::div(phi, U);
        auto exprOpt = dsl::optimize(expr);
        exprOpt.read(input);

        BENCHMARK(std::string(execName)) { return exprOpt.assemble(mesh, 1.0, 1.0); };
    }
}

}
