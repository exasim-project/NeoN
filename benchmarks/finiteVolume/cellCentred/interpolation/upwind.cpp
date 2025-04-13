// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "../../../catch_main.hpp"

#include <catch2/catch_template_test_macros.hpp>

using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;
using NeoN::Input;

namespace NeoN
{

TEMPLATE_TEST_CASE("upwind", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    UnstructuredMesh mesh = create1DUniformMesh(exec, size);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);
    Input input = TokenList({std::string("upwind")});
    auto upwind = SurfaceInterpolation<TestType>(exec, mesh, input);

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto flux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, surfaceBCs);

    fill(flux.internalVector(), one<scalar>());
    fill(in.internalVector(), one<TestType>());

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName)) { return (upwind.interpolate(flux, in, out)); };
    }
}

}
