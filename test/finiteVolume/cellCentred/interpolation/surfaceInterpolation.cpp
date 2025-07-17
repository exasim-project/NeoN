// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("SurfaceInterpolation")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    std::string interpolation = GENERATE(std::string("linear"), std::string("upwind"));

    SECTION("Construct from Token" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        NeoN::Input input = NeoN::TokenList({interpolation});
        fvcc::SurfaceInterpolation<NeoN::scalar> surfInterpolation(exec, mesh, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        NeoN::Input input =
            NeoN::Dictionary({{std::string("surfaceInterpolation"), interpolation}});
        fvcc::SurfaceInterpolation<NeoN::scalar> surfInterpolation(exec, mesh, input);
    }
}
