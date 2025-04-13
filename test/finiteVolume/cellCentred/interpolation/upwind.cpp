// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

namespace NeoN
{

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("upwind", "", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    Input input = TokenList({std::string("upwind")});
    auto upwind = SurfaceInterpolation<TestType>(exec, mesh, input);
    std::vector<fvcc::SurfaceBoundary<TestType>> bcs {};
    for (auto patchi : I<size_t> {0, 1})
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", one<TestType>());
        bcs.push_back(fvcc::SurfaceBoundary<TestType>(mesh, dict, patchi));
    }

    auto in = VolumeField<TestType>(exec, "in", mesh, {});
    auto flux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, bcs);

    fill(flux.internalVector(), one<scalar>());
    fill(in.internalVector(), one<TestType>());

    upwind.interpolate(flux, in, out);
    out.correctBoundaryConditions();

    auto outHost = out.internalVector().copyToHost();
    auto nInternal = mesh.nInternalFaces();
    auto nBoundary = mesh.nBoundaryFaces();
    for (int i = 0; i < nInternal; i++)
    {
        REQUIRE(outHost.view()[i] == one<TestType>());
    }

    for (int i = nInternal; i < nInternal + nBoundary; i++)
    {
        REQUIRE(outHost.view()[i] == one<TestType>());
    }
}

}
