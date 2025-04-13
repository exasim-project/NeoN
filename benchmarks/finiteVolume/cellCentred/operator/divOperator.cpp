// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

using Operator = NeoN::dsl::Operator;

TEST_CASE("DivOperator::div", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceVector<NeoN::scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    NeoN::fill(faceFlux.internalVector(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
    fvcc::VolumeVector<NeoN::scalar> phi(exec, "vf", mesh, volumeBCs);
    fvcc::VolumeVector<NeoN::scalar> divPhi(exec, "divPhi", mesh, volumeBCs);
    NeoN::fill(phi.internalVector(), 1.0);

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        NeoN::Input input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);

        BENCHMARK(std::string(execName)) { return (op.div(divPhi)); };
    }
}
