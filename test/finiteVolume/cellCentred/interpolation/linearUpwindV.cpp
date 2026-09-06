// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <algorithm>

using NeoN::finiteVolume::cellCentred::SurfaceField;
using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;

namespace NeoN
{

// linearUpwindV is linearUpwind whose deferred gradient correction is built from the
// cell-limited (minmod, k=1) gradient instead of the unlimited Gauss-Green one. Two properties
// follow, and both are pinned below:
//   * where the limiter is inactive (a linear field) it must agree with linearUpwind exactly;
//   * where it is active it must keep the reconstructed face value inside the value range
//     spanned by the upwind cell and its neighbours -- that bound is the point of the scheme.

// The scheme is a vector reconstruction, so it is registered only for Vec3. A missing
// registration aborts the process at lookup rather than throwing, so assert on the factory
// table instead of trying to construct and catch.
TEST_CASE("linearUpwindV is registered for vector fields only")
{
    const auto vectorEntries = fvcc::SurfaceInterpolationFactory<Vec3>::entries();
    const auto scalarEntries = fvcc::SurfaceInterpolationFactory<scalar>::entries();

    auto has = [](const auto& entries, const std::string& key)
    { return std::find(entries.begin(), entries.end(), key) != entries.end(); };

    REQUIRE(has(vectorEntries, "linearUpwindV"));
    REQUIRE(has(vectorEntries, "linearUpwind"));
    // Cell limiting is a vector-field concept; the scalar factory must not offer it.
    REQUIRE_FALSE(has(scalarEntries, "linearUpwindV"));
}

// A field whose gradient is constant is monotone across every cell, so the minmod limiter never
// clips: linearUpwindV must reproduce the analytic face value exactly, just like linearUpwind.
// Only interior-interior faces are checked, where the Gauss gradient itself is exact.
TEST_CASE("linearUpwindV is exact for a linear vector field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    const scalar l = 1.0;
    const scalar h = l / static_cast<scalar>(n);
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);

    // Capture geometry on the host before any scheme construction frees the mesh centres.
    auto ccH = mesh.cellCenters().copyToHost();
    auto fcH = mesh.faceCenters().copyToHost();
    auto ownH = mesh.faceOwners().copyToHost();
    auto neiH = mesh.faceNeighbors().copyToHost();
    const auto cc = ccH.view();
    const auto fc = fcH.view();
    const auto own = ownH.view();
    const auto nei = neiH.view();

    auto f = [&](const Vec3& p)
    {
        return Vec3 {
            1.5 * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3,
            -0.5 * p[0] + 1.25 * p[1] + 0.5 * p[2] - 0.2,
            0.25 * p[0] + 0.5 * p[1] - 1.75 * p[2] + 0.1
        };
    };

    auto interior = [&](localIdx celli)
    {
        const Vec3 p = cc[celli];
        for (std::size_t dd = 0; dd < 3; ++dd)
        {
            if (p[dd] < h || p[dd] > l - h) return false;
        }
        return true;
    };

    auto src = VolumeField<Vec3>(
        exec, "src", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh)
    );
    {
        auto hostIn = src.internalVector().copyToHost();
        for (localIdx i = 0; i < hostIn.size(); ++i)
            hostIn.view()[i] = f(cc[i]);
        src.internalVector() = hostIn.copyToExecutor(exec);
    }

    Input input = TokenList({std::string("linearUpwindV"), std::string("Gauss")});
    auto scheme = SurfaceInterpolation<Vec3>(exec, mesh, input);
    auto flux = SurfaceField<scalar>(
        exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    auto out = SurfaceField<Vec3>(
        exec, "out", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Vec3>>(mesh)
    );

    auto checkDirection = [&](scalar fluxSign)
    {
        fill(flux.internalVector(), fluxSign);
        scheme.interpolate(flux, src, out);
        auto outH = out.internalVector().copyToHost();
        const auto outV = outH.view();

        localIdx nChecked = 0;
        for (localIdx facei = 0; facei < mesh.nInternalFaces(); ++facei)
        {
            if (!interior(own[facei]) || !interior(nei[facei])) continue;
            const Vec3 expected = f(fc[facei]);
            for (std::size_t cmpt = 0; cmpt < 3; ++cmpt)
            {
                REQUIRE(outV[facei][cmpt] == Catch::Approx(expected[cmpt]).margin(1e-11));
            }
            ++nChecked;
        }
        REQUIRE(nChecked > 0);
    };

    checkDirection(1.0);  // upwind = owner
    checkDirection(-1.0); // upwind = neighbour
}

// With a sharp peak the unlimited gradient extrapolates past the local data range. The
// cell-limited gradient clips it, so every reconstructed face value must stay inside the range
// spanned by the upwind cell and its face neighbours. The test also asserts that the two schemes
// actually disagree somewhere, which is what proves the cell-limited path is being taken rather
// than silently falling through to plain linearUpwind.
TEST_CASE("linearUpwindV bounds the reconstruction at a sharp extremum")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 5;
    const scalar l = 1.0;
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);

    auto ownH = mesh.faceOwners().copyToHost();
    auto neiH = mesh.faceNeighbors().copyToHost();
    const auto own = ownH.view();
    const auto nei = neiH.view();
    const auto nCells = mesh.nCells();
    const auto nInt = mesh.nInternalFaces();

    // A single-cell spike: zero everywhere except one interior cell.
    const localIdx spike = nCells / 2;
    std::vector<Vec3> values(static_cast<std::size_t>(nCells), Vec3 {0.0, 0.0, 0.0});
    values[static_cast<std::size_t>(spike)] = Vec3 {10.0, -8.0, 6.0};

    auto src = VolumeField<Vec3>(
        exec, "src", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh)
    );
    src.internalVector() = Vector<Vec3>(exec, values);
    src.correctBoundaryConditions();

    // Per-cell min/max over the cell itself and its face neighbours -- the range the cell-limited
    // reconstruction is allowed to produce on that cell's faces.
    std::vector<Vec3> lo = values;
    std::vector<Vec3> hi = values;
    for (localIdx facei = 0; facei < nInt; ++facei)
    {
        const auto o = static_cast<std::size_t>(own[facei]);
        const auto nb = static_cast<std::size_t>(nei[facei]);
        for (std::size_t cmpt = 0; cmpt < 3; ++cmpt)
        {
            lo[o][cmpt] = std::min(lo[o][cmpt], values[nb][cmpt]);
            hi[o][cmpt] = std::max(hi[o][cmpt], values[nb][cmpt]);
            lo[nb][cmpt] = std::min(lo[nb][cmpt], values[o][cmpt]);
            hi[nb][cmpt] = std::max(hi[nb][cmpt], values[o][cmpt]);
        }
    }

    auto interpolateWith = [&](const std::string& schemeName)
    {
        Input input = TokenList({schemeName, std::string("Gauss")});
        auto scheme = SurfaceInterpolation<Vec3>(exec, mesh, input);
        auto flux = SurfaceField<scalar>(
            exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
        );
        auto out = SurfaceField<Vec3>(
            exec, "out", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Vec3>>(mesh)
        );
        fill(flux.internalVector(), one<scalar>()); // upwind = owner on every internal face
        scheme.interpolate(flux, src, out);
        return out.internalVector().copyToHost();
    };

    auto limitedH = interpolateWith("linearUpwindV");
    auto plainH = interpolateWith("linearUpwind");
    const auto limited = limitedH.view();
    const auto plain = plainH.view();

    bool differs = false;
    for (localIdx facei = 0; facei < nInt; ++facei)
    {
        const auto o = static_cast<std::size_t>(own[facei]);
        for (std::size_t cmpt = 0; cmpt < 3; ++cmpt)
        {
            REQUIRE(limited[facei][cmpt] >= lo[o][cmpt] - 1e-10);
            REQUIRE(limited[facei][cmpt] <= hi[o][cmpt] + 1e-10);
            if (std::abs(limited[facei][cmpt] - plain[facei][cmpt]) > 1e-10) differs = true;
        }
    }
    // The limiter must actually bite somewhere on this field.
    REQUIRE(differs);
}

}
