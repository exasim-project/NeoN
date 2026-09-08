// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

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

namespace
{

// 1D uniform mesh of nCells with a unit flux in the +x direction, so the owner of every internal
// face is the upwind cell. Returns the interpolated face values of the given cell values.
Vector<scalar> interpolateOnRamp(
    const Executor& exec,
    const std::vector<scalar>& cellValues,
    const scalar boundaryValue,
    const scalar k
)
{
    auto mesh = create1DUniformMesh(exec, static_cast<localIdx>(cellValues.size()));
    Input input = TokenList({std::string("limitedLinear"), k});
    auto scheme = SurfaceInterpolation<scalar>(exec, mesh, input);

    std::vector<fvcc::VolumeBoundary<scalar>> vbcs {};
    std::vector<fvcc::SurfaceBoundary<scalar>> sbcs {};
    for (auto patchi : I<localIdx> {0, 1})
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", boundaryValue);
        vbcs.push_back(fvcc::VolumeBoundary<scalar>(mesh, dict, patchi));
        sbcs.push_back(fvcc::SurfaceBoundary<scalar>(mesh, dict, patchi));
    }

    auto in = VolumeField<scalar>(exec, "in", mesh, vbcs);
    auto flux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<scalar>(exec, "out", mesh, sbcs);

    in.internalVector() = Vector<scalar>(exec, cellValues);
    in.correctBoundaryConditions();
    fill(flux.internalVector(), one<scalar>());

    scheme.interpolate(flux, in, out);
    return out.internalVector().copyToHost();
}

} // namespace

// A constant field has a zero face gradient, so the ratio saturates, the limiter is 1 and the
// scheme is pure central differencing — which reproduces the constant exactly.
TEST_CASE("limitedLinear preserves a constant field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const std::vector<scalar> cellValues(10, 1.0);
    auto outHost = interpolateOnRamp(exec, cellValues, 1.0, 1.0);

    for (localIdx i = 0; i < outHost.size(); i++)
    {
        REQUIRE(outHost.view()[i] == Catch::Approx(1.0).margin(1e-12));
    }
}

// On a smooth (linear) profile the upwind-cell gradient equals the face gradient, so r == 1, the
// limiter is 1 and the scheme must fall back to second-order central differencing: on a uniform
// mesh the face value is the average of the two adjacent cell values. Faces adjacent to a
// boundary cell are excluded — the boundary condition, not the ramp, sets that cell's gradient.
TEST_CASE("limitedLinear is central differencing on a smooth profile")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    std::vector<scalar> cellValues(10);
    for (std::size_t i = 0; i < cellValues.size(); i++)
    {
        cellValues[i] = static_cast<scalar>(i);
    }
    auto outHost = interpolateOnRamp(exec, cellValues, 0.0, 1.0);

    for (localIdx i = 1; i < 8; i++)
    {
        const scalar expected = 0.5
                              * (cellValues[static_cast<std::size_t>(i)]
                                 + cellValues[static_cast<std::size_t>(i) + 1]);
        REQUIRE(outHost.view()[i] == Catch::Approx(expected).margin(1e-12));
    }
}

// At a local extremum r <= 0, so the limiter collapses to 0 and the scheme is pure upwind. This is
// the boundedness property limitedLinear exists for: the face values either side of a single-cell
// spike must not overshoot it.
TEST_CASE("limitedLinear reduces to upwind at a local extremum")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    std::vector<scalar> cellValues(10, 0.0);
    cellValues[5] = 1.0;
    auto outHost = interpolateOnRamp(exec, cellValues, 0.0, 1.0);

    // Flux is positive, so the owner (lower cell index) is upwind on every face.
    REQUIRE(outHost.view()[4] == Catch::Approx(0.0).margin(1e-12)); // upwind cell 4 -> 0
    REQUIRE(outHost.view()[5] == Catch::Approx(1.0).margin(1e-12)); // upwind cell 5 -> 1

    // No face may exceed the range of the cell values it was built from.
    for (localIdx i = 0; i < outHost.size(); i++)
    {
        REQUIRE(outHost.view()[i] >= -1e-12);
        REQUIRE(outHost.view()[i] <= 1.0 + 1e-12);
    }
}

// The limitedLinear coefficient is restricted to [0, 1]; anything else is a spec error and must be
// rejected at construction rather than silently producing an unbounded limiter.
TEST_CASE("limitedLinear rejects an out-of-range coefficient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create1DUniformMesh(exec, 10);
    Input tooLarge = TokenList({std::string("limitedLinear"), static_cast<scalar>(2.0)});
    REQUIRE_THROWS_AS(SurfaceInterpolation<scalar>(exec, mesh, tooLarge), NeoN::NeoNException);

    Input negative = TokenList({std::string("limitedLinear"), static_cast<scalar>(-0.5)});
    REQUIRE_THROWS_AS(SurfaceInterpolation<scalar>(exec, mesh, negative), NeoN::NeoNException);
}

// The coefficient is part of the scheme spec, not an optional tweak: without it the scheme would
// silently run with a different limiter than the one the case asked for.
TEST_CASE("limitedLinear rejects a missing coefficient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create1DUniformMesh(exec, 10);
    Input noCoeff = TokenList({std::string("limitedLinear")});
    REQUIRE_THROWS_AS(SurfaceInterpolation<scalar>(exec, mesh, noCoeff), NeoN::NeoNException);

    Input wrongType = TokenList({std::string("limitedLinear"), std::string("one")});
    REQUIRE_THROWS_AS(SurfaceInterpolation<scalar>(exec, mesh, wrongType), NeoN::NeoNException);

    Dictionary emptyDict;
    emptyDict.insert("surfaceInterpolation", std::string("limitedLinear"));
    REQUIRE_THROWS_AS(
        SurfaceInterpolation<scalar>(exec, mesh, Input(emptyDict)), NeoN::NeoNException
    );
}

} // namespace NeoN
