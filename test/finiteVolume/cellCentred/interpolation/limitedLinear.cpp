// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::toLocalIdx;
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
    const scalar k,
    const bool cellLimitedGrad = false
)
{
    auto mesh = create1DUniformMesh(exec, static_cast<localIdx>(cellValues.size()));
    Input input = cellLimitedGrad
                    ? TokenList({std::string("limitedLinear"), k, std::string("cellLimited")})
                    : TokenList({std::string("limitedLinear"), k});
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

    for (localIdx i = 0; i < toLocalIdx(outHost.size()); i++)
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
    for (sizeType i = 0; i < cellValues.size(); i++)
    {
        cellValues[i] = static_cast<scalar>(i);
    }
    auto outHost = interpolateOnRamp(exec, cellValues, 0.0, 1.0);

    for (localIdx i = 1; i < 8; i++)
    {
        const scalar expected =
            0.5 * (cellValues[static_cast<sizeType>(i)] + cellValues[static_cast<sizeType>(i) + 1]);
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
    for (localIdx i = 0; i < toLocalIdx(outHost.size()); i++)
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

// The limiter gradient is selectable because it belongs to the transported field rather than to
// this scheme, and a cell-limited one reports a smaller upwind-cell slope than the Gauss-Green
// default. Smaller slope -> smaller TVD ratio -> smaller limiter, so the blend leans further
// towards upwind and can never overshoot what the unlimited gradient produces.
TEST_CASE("limitedLinear cellLimited leans further towards upwind")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    // Monotone increasing, but with a step sharp enough that extrapolating the unlimited gradient
    // from cell 4 undershoots its neighbour minimum -- which is exactly what cellLimited clips.
    const std::vector<scalar> cellValues {0.0, 0.0, 0.0, 0.0, 0.2, 2.0, 2.0, 2.0, 2.0, 2.0};

    auto plain = interpolateOnRamp(exec, cellValues, 0.0, 1.0, false);
    auto limited = interpolateOnRamp(exec, cellValues, 0.0, 1.0, true);

    // Flux is +x, so the owner (lower index) is upwind on every face and central differencing sits
    // above the upwind value on this monotone profile.
    bool differs = false;
    for (localIdx i = 0; i < toLocalIdx(plain.size()); i++)
    {
        REQUIRE(limited.view()[i] <= plain.view()[i] + 1e-12);
        if (limited.view()[i] < plain.view()[i] - 1e-12) differs = true;
    }
    // Guard against a vacuous pass: the flag has to change something on this profile.
    REQUIRE(differs);

    // Face 4 separates cells 4 and 5. The clipped gradient drives the limiter to zero there, so
    // the face takes the upwind cell value outright.
    REQUIRE(limited.view()[4] == Catch::Approx(cellValues[4]).margin(1e-12));
}

// An unrecognised word after the coefficient is a spec error: silently ignoring it would run the
// case with the default gradient while the spec asked for something else.
TEST_CASE("limitedLinear rejects an unknown gradient marker")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create1DUniformMesh(exec, 10);
    Input bogus = TokenList(
        {std::string("limitedLinear"), static_cast<scalar>(1.0), std::string("leastSquares")}
    );
    REQUIRE_THROWS_AS(SurfaceInterpolation<scalar>(exec, mesh, bogus), NeoN::NeoNException);
}

namespace
{

// Face values of limitedLinearV on a 1D uniform mesh (unit cells, unit faces) for the given vector
// cell values and a uniform face flux of `flux`, with a fixedValue boundary of `boundaryValue`.
Vector<Vec3> interpolateVectorOnLine(
    const Executor& exec,
    const std::vector<Vec3>& cellValues,
    const Vec3& boundaryValue,
    const scalar k,
    const scalar flux
)
{
    auto mesh = create1DUniformMesh(exec, static_cast<localIdx>(cellValues.size()));
    Input input = TokenList({std::string("limitedLinearV"), k});
    auto scheme = SurfaceInterpolation<Vec3>(exec, mesh, input);

    std::vector<fvcc::VolumeBoundary<Vec3>> vbcs {};
    std::vector<fvcc::SurfaceBoundary<Vec3>> sbcs {};
    for (auto patchi : I<localIdx> {0, 1})
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", boundaryValue);
        vbcs.push_back(fvcc::VolumeBoundary<Vec3>(mesh, dict, patchi));
        sbcs.push_back(fvcc::SurfaceBoundary<Vec3>(mesh, dict, patchi));
    }

    auto in = VolumeField<Vec3>(exec, "in", mesh, vbcs);
    auto faceFlux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<Vec3>(exec, "out", mesh, sbcs);

    in.internalVector() = Vector<Vec3>(exec, cellValues);
    in.correctBoundaryConditions();
    fill(faceFlux.internalVector(), flux);

    scheme.interpolate(faceFlux, in, out);
    return out.internalVector().copyToHost();
}

// OpenFOAM's limitedLinear limiter for a ratio r: max(min(2/k * r, 1), 0), with r from the slopes
// gradf along the face and gradcf of the upwind cell (the 1000 cap of NVDTVD / NVDVTVDV).
scalar referenceLimiter(const scalar k, const scalar gradf, const scalar gradcf)
{
    const scalar sign = [](scalar x) { return x >= 0 ? scalar(1) : scalar(-1); }(gradcf) *
                        [](scalar x) { return x >= 0 ? scalar(1) : scalar(-1); }(gradf);
    const scalar r =
        std::abs(gradcf) >= 1000 * std::abs(gradf) ? 2 * 1000 * sign - 1 : 2 * (gradcf / gradf) - 1;
    return std::max(std::min(2 / k * r, scalar(1)), scalar(0));
}

} // namespace

// limitedLinearV is OpenFOAM's NVDVTVDV: the TVD ratio is formed along the face difference of the
// whole vector, gradf = dU & dU and gradcf = dU & (d & grad(U)_upwind), so ONE limiter serves the
// three components. The expectation is written out by hand for interior faces of a 1D line, where
// the Gauss gradient of an interior cell is (U[i+1] - U[i-1]) / 2, and it is checked against the
// per-component limiter to prove the test can tell the two apart.
TEST_CASE("limitedLinearV limits the vector along its face difference")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const std::vector<Vec3> cells {
        {0.0, 0.0, 0.0},
        {1.0, 0.5, 0.0},
        {1.5, -1.0, 0.2},
        {1.0, -1.2, 0.1},
        {2.5, 0.3, -0.4},
        {3.0, 1.1, 0.0},
        {2.2, 1.3, 0.6},
        {2.0, 0.4, 0.5},
        {2.1, 0.2, 0.4},
        {0.0, 0.0, 0.0},
    };
    const auto n = static_cast<localIdx>(cells.size());

    for (const scalar flux : {1.0, -1.0})
    {
        for (const scalar k : {1.0, 0.5})
        {
            INFO("flux " << flux << ", k " << k);
            auto out = interpolateVectorOnLine(exec, cells, Vec3 {0.0, 0.0, 0.0}, k, flux);

            bool differsFromPerComponent = false;
            bool limiterStrictlyBetween = false;
            // faces whose upwind cell is an interior cell and whose neighbour gradient needs no BC
            for (localIdx i = 1; i < n - 2; i++)
            {
                const auto own = static_cast<sizeType>(i);
                const auto nei = own + 1;
                const auto up = flux > 0 ? own : nei;
                const Vec3 gradUp = (cells[up + 1] - cells[up - 1]) * 0.5;
                const Vec3 dU = cells[nei] - cells[own];

                const scalar limiter = referenceLimiter(k, dU & dU, dU & gradUp);
                const scalar upwindW = flux >= 0 ? 1.0 : 0.0;
                const scalar w = limiter * 0.5 + (1.0 - limiter) * upwindW;
                const Vec3 expected = cells[own] * w + cells[nei] * (1.0 - w);
                for (int c = 0; c < 3; c++)
                {
                    REQUIRE(out.view()[i][c] == Catch::Approx(expected[c]).margin(1e-12));
                }

                limiterStrictlyBetween =
                    limiterStrictlyBetween || (limiter > 0.05 && limiter < 0.95);
                for (int c = 0; c < 3; c++)
                {
                    const scalar lc = referenceLimiter(k, dU[c], gradUp[c]);
                    const scalar wc = lc * 0.5 + (1.0 - lc) * upwindW;
                    const scalar perComponent = cells[own][c] * wc + cells[nei][c] * (1.0 - wc);
                    differsFromPerComponent =
                        differsFromPerComponent || std::abs(perComponent - expected[c]) > 1e-6;
                }
            }
            // The profile has to exercise a partial limiter and one a per-component limiter gets
            // wrong, or the checks above would also pass for the wrong scheme.
            REQUIRE(limiterStrictlyBetween);
            REQUIRE(differsFromPerComponent);
        }
    }
}

// A constant vector has a zero face difference and a zero gradient: the ratio saturates, the
// limiter is 1 and the scheme is central differencing, which reproduces the constant.
TEST_CASE("limitedLinearV preserves a constant vector field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const Vec3 value {1.0, -2.0, 0.5};
    const std::vector<Vec3> cells(10, value);
    auto out = interpolateVectorOnLine(exec, cells, value, 1.0, 1.0);

    for (localIdx i = 0; i < toLocalIdx(out.size()); i++)
    {
        for (int c = 0; c < 3; c++)
        {
            REQUIRE(out.view()[i][c] == Catch::Approx(value[c]).margin(1e-12));
        }
    }
}

// The cell-limited gradient exists for the scalar limiter only: asking limitedLinearV for it is a
// spec error rather than a silent fallback to the unlimited gradient.
TEST_CASE("limitedLinearV rejects the cellLimited marker")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create1DUniformMesh(exec, 10);
    Input marked = TokenList(
        {std::string("limitedLinearV"), static_cast<scalar>(1.0), std::string("cellLimited")}
    );
    REQUIRE_THROWS_AS(SurfaceInterpolation<Vec3>(exec, mesh, marked), NeoN::NeoNException);
}

} // namespace NeoN
