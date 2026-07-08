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

// A constant field has zero gradient, so the linearUpwind correction vanishes and the scheme
// must reduce to plain upwind. Mirrors the upwind interpolation test.
TEMPLATE_TEST_CASE("linearUpwind reduces to upwind for a constant field", "", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create1DUniformMesh(exec, 10);
    Input input = TokenList({std::string("linearUpwind"), std::string("Gauss")});
    auto linearUpwind = SurfaceInterpolation<TestType>(exec, mesh, input);

    std::vector<fvcc::VolumeBoundary<TestType>> vbcs {};
    std::vector<fvcc::SurfaceBoundary<TestType>> sbcs {};
    for (auto patchi : I<localIdx> {0, 1})
    {
        Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", one<TestType>());
        vbcs.push_back(fvcc::VolumeBoundary<TestType>(mesh, dict, patchi));
        sbcs.push_back(fvcc::SurfaceBoundary<TestType>(mesh, dict, patchi));
    }

    auto in = VolumeField<TestType>(exec, "in", mesh, vbcs);
    auto flux = SurfaceField<scalar>(exec, "flux", mesh, {});
    auto out = SurfaceField<TestType>(exec, "out", mesh, sbcs);

    fill(flux.internalVector(), one<scalar>());
    fill(in.internalVector(), one<TestType>());
    in.correctBoundaryConditions();

    linearUpwind.interpolate(flux, in, out);
    out.correctBoundaryConditions();

    auto outHost = out.internalVector().copyToHost();
    for (localIdx i = 0; i < mesh.nInternalFaces(); i++)
    {
        REQUIRE(outHost.view()[i] == one<TestType>());
    }

    auto outBHost = out.boundaryData().value().copyToHost();
    for (localIdx i = 0; i < mesh.nBoundaryFaces(); i++)
    {
        REQUIRE(outBHost.view()[i] == one<TestType>());
    }
}

// The geometry scheme must cache, per internal face, the vector from each adjacent cell centre to
// the face centre. On an orthogonal uniform mesh with spacing h these have magnitude h/2 and are
// exact opposites (the face centre sits midway between the two cell centres).
TEST_CASE("GeometryScheme caches face-to-cell delta vectors")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    const scalar l = 1.0;
    const scalar h = l / static_cast<scalar>(n);
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);

    fvcc::GeometryScheme scheme(exec, mesh, std::make_unique<fvcc::BasicGeometryScheme>(mesh));

    auto dOwn = scheme.faceDeltaOwner().internalVector().copyToHost();
    auto dNei = scheme.faceDeltaNeighbour().internalVector().copyToHost();
    const auto dOwnV = dOwn.view();
    const auto dNeiV = dNei.view();

    REQUIRE(dOwn.size() == mesh.nInternalFaces());
    for (localIdx i = 0; i < dOwn.size(); ++i)
    {
        REQUIRE(mag(dOwnV[i]) == Catch::Approx(0.5 * h).margin(1e-12));
        REQUIRE(mag(dNeiV[i]) == Catch::Approx(0.5 * h).margin(1e-12));
        // Cf - C_own == -(Cf - C_nei) on an orthogonal uniform mesh
        REQUIRE(mag(dOwnV[i] + dNeiV[i]) < 1e-12);
    }
}

// linearUpwind reconstructs a field exactly where its gradient is exact. For a linear field on a
// uniform mesh the Gauss gradient is exact at every interior cell (all its faces are internal and
// linear interpolation of a linear field is exact), so for any internal face between two interior
// cells the reconstructed face value must equal the analytic field at the face centre — for both
// upwind directions.
TEST_CASE("linearUpwind is exact for a linear scalar field")
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

    // analytic linear field f(x) = a.x + d
    const Vec3 a {1.5, -2.0, 0.75};
    const scalar d = 0.3;
    auto f = [&](const Vec3& p) { return a[0] * p[0] + a[1] * p[1] + a[2] * p[2] + d; };

    auto interior = [&](localIdx celli)
    {
        const Vec3 p = cc[celli];
        for (std::size_t dd = 0; dd < 3; ++dd)
        {
            if (p[dd] < h || p[dd] > l - h) return false;
        }
        return true;
    };

    auto src = VolumeField<scalar>(
        exec, "src", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh)
    );
    {
        auto hostIn = src.internalVector().copyToHost();
        for (localIdx i = 0; i < hostIn.size(); ++i)
            hostIn.view()[i] = f(cc[i]);
        src.internalVector() = hostIn.copyToExecutor(exec);
    }

    Input input = TokenList({std::string("linearUpwind"), std::string("Gauss")});
    auto linearUpwind = SurfaceInterpolation<scalar>(exec, mesh, input);
    auto flux = SurfaceField<scalar>(
        exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    auto out = SurfaceField<scalar>(
        exec, "out", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );

    auto checkDirection = [&](scalar fluxSign)
    {
        fill(flux.internalVector(), fluxSign);
        linearUpwind.interpolate(flux, src, out);
        auto outH = out.internalVector().copyToHost();
        const auto outV = outH.view();

        localIdx nChecked = 0;
        for (localIdx facei = 0; facei < mesh.nInternalFaces(); ++facei)
        {
            if (!interior(own[facei]) || !interior(nei[facei])) continue;
            REQUIRE(outV[facei] == Catch::Approx(f(fc[facei])).margin(1e-11));
            ++nChecked;
        }
        REQUIRE(nChecked > 0);
    };

    checkDirection(1.0);  // upwind = owner
    checkDirection(-1.0); // upwind = neighbour
}

// Same exactness property for a linear vector field: the Vec3 reconstruction uses the cell tensor
// gradient and must match the analytic field at the face centre on interior-interior faces.
TEST_CASE("linearUpwind is exact for a linear vector field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const localIdx n = 4;
    const scalar l = 1.0;
    const scalar h = l / static_cast<scalar>(n);
    auto mesh = create3DUniformMesh(exec, n, n, n, l, l, l);

    auto ccH = mesh.cellCenters().copyToHost();
    auto fcH = mesh.faceCenters().copyToHost();
    auto ownH = mesh.faceOwners().copyToHost();
    auto neiH = mesh.faceNeighbors().copyToHost();
    const auto cc = ccH.view();
    const auto fc = fcH.view();
    const auto own = ownH.view();
    const auto nei = neiH.view();

    // analytic linear vector field U(x) with a distinct linear law per component
    auto f = [&](const Vec3& p)
    {
        return Vec3 {
            1.5 * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3,
            -0.5 * p[0] + 1.25 * p[1] - 0.4 * p[2] - 0.1,
            2.0 * p[0] + 0.6 * p[1] - 1.1 * p[2] + 0.2
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

    Input input = TokenList({std::string("linearUpwind"), std::string("Gauss")});
    auto linearUpwind = SurfaceInterpolation<Vec3>(exec, mesh, input);
    auto flux = SurfaceField<scalar>(
        exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    auto out = SurfaceField<Vec3>(
        exec, "out", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<Vec3>>(mesh)
    );

    auto checkDirection = [&](scalar fluxSign)
    {
        fill(flux.internalVector(), fluxSign);
        linearUpwind.interpolate(flux, src, out);
        auto outH = out.internalVector().copyToHost();
        const auto outV = outH.view();

        localIdx nChecked = 0;
        for (localIdx facei = 0; facei < mesh.nInternalFaces(); ++facei)
        {
            if (!interior(own[facei]) || !interior(nei[facei])) continue;
            const Vec3 expected = f(fc[facei]);
            const Vec3 got = outV[facei];
            REQUIRE(mag(got - expected) < 1e-11);
            ++nChecked;
        }
        REQUIRE(nChecked > 0);
    };

    checkDirection(1.0);
    checkDirection(-1.0);
}

}
