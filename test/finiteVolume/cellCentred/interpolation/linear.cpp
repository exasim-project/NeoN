// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

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

TEMPLATE_TEST_CASE("linear", "", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    Input input = TokenList({std::string("linear")});
    auto linear = SurfaceInterpolation<TestType>(exec, mesh, input);
    std::vector<fvcc::VolumeBoundary<TestType>> vbcs {};
    std::vector<fvcc::SurfaceBoundary<TestType>> sbcs {};
    for (auto patchi : I<size_t> {0, 1})
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", one<TestType>());
        sbcs.push_back(fvcc::SurfaceBoundary<TestType>(mesh, dict, patchi));
        vbcs.push_back(fvcc::VolumeBoundary<TestType>(mesh, dict, patchi));
    }

    auto in = VolumeField<TestType>(exec, "in", mesh, vbcs);
    auto out = SurfaceField<TestType>(exec, "out", mesh, sbcs);

    fill(in.internalField(), one<TestType>());
    in.correctBoundaryConditions();

    linear.interpolate(in, out);
    out.correctBoundaryConditions();

    auto outHost = out.internalField().copyToHost();
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
