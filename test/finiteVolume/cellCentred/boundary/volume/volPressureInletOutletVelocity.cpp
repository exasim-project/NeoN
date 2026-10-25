// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

// Build a face-flux phi (surface scalar field) whose boundary faces all carry `flux`. Mirrors
// volInletOutlet.cpp's own makeFlux helper.
fvcc::SurfaceField<NeoN::scalar>
makeFlux(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::scalar flux)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> phi(exec, "phi", mesh, bcs);
    NeoN::fill(phi.internalVector(), flux);
    NeoN::fill(phi.boundaryData().value(), flux);
    return phi;
}

TEST_CASE("pressureInletOutletVelocity_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Single-cell mesh (createSingleCellMesh): patch 0's unit outward normal is (-1, 0, 0),
    // magSf = 1. Owner-cell U = (2, 3, 4), tangentialVelocity dict entry = (3, 5, 7).
    //   n & ownerValue             = -2
    //   n*(n & ownerValue)         = (2, 0, 0)            -- recovers (U_x, 0, 0)
    //   n & tangentialVelocity     = -3
    //   tangentialVelocityProjected = (3,5,7) - (-1,0,0)*(-3) = (0, 5, 7)
    //   refValue = n*(n & ownerValue) + tangentialVelocityProjected = (2, 5, 7)
    // On inflow (valueFraction=1): value = refValue = (2, 5, 7).
    // On outflow (valueFraction=0): value = ownerValue = (2, 3, 4).
    // This matches OpenFOAM's tensor-valued directionMixed blend exactly (see
    // pressureInletOutletVelocity.hpp's own derivation): the normal component always tracks the
    // owner cell (letting the pressure equation set it), the tangential components switch
    // between the specified tangentialVelocity (inflow) and zero-gradient (outflow).
    auto mesh = NeoN::createSingleCellMesh(exec);
    const NeoN::Vec3 ownerValue(2.0, 3.0, 4.0);
    const NeoN::Vec3 tangentialVelocity(3.0, 5.0, 7.0);
    const NeoN::Vec3 expectedRefValue(2.0, 5.0, 7.0);

    NeoN::Dictionary dict;
    dict.insert("type", std::string("pressureInletOutletVelocity"));
    dict.insert("tangentialVelocity", tangentialVelocity);
    auto boundary = fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create(
        "pressureInletOutletVelocity", mesh, dict, 0
    );

    // fixesValue=false: mixed BC, not a global Dirichlet patch.
    REQUIRE(boundary->attributes().fixesValue == false);

    SECTION("inflow (phi < 0): tangential-plane Dirichlet, normal tracks owner cell " + execName)
    {
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        auto phi = makeFlux(exec, mesh, -2.0);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().refGrad(),
            field.boundaryData().valueFraction()
        );

        for (auto& v : valuesH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(v[d] == Catch::Approx(expectedRefValue[d]));
        }
        for (auto& v : refValuesH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(v[d] == Catch::Approx(expectedRefValue[d]));
        }
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(1.0));
        for (auto& g : refGradH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(g[d] == Catch::Approx(0.0));
        }
    }

    SECTION("outflow (phi >= 0): zero-gradient on all 3 components " + execName)
    {
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        auto phi = makeFlux(exec, mesh, 2.0);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(v[d] == Catch::Approx(ownerValue[d]));
        }
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(0.0));
    }

    SECTION("no phi in context => degenerates to zero-gradient " + execName)
    {
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        fvcc::BoundaryContext ctx; // deliberately empty

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(v[d] == Catch::Approx(ownerValue[d]));
        }
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(0.0));
    }

    SECTION("no context at all => degenerates to zero-gradient " + execName)
    {
        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        boundary->correctBoundaryCondition(field);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(boundary->range()))
        {
            for (auto d = 0u; d < 3; ++d)
                REQUIRE(v[d] == Catch::Approx(ownerValue[d]));
        }
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(0.0));
    }

    SECTION("default tangentialVelocity is zero " + execName)
    {
        NeoN::Dictionary defaultDict;
        defaultDict.insert("type", std::string("pressureInletOutletVelocity"));
        auto defaultBoundary = fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create(
            "pressureInletOutletVelocity", mesh, defaultDict, 0
        );

        auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        auto phi = makeFlux(exec, mesh, -2.0);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);

        defaultBoundary->correctBoundaryCondition(field, ctx);

        // n*(n & ownerValue) = (2,0,0), tangentialVelocity = 0 => refValue = (2, 0, 0).
        auto valuesH = copyToHosts(field.boundaryData().value());
        for (auto& v : std::get<0>(valuesH).view(defaultBoundary->range()))
        {
            REQUIRE(v[0] == Catch::Approx(2.0));
            REQUIRE(v[1] == Catch::Approx(0.0));
            REQUIRE(v[2] == Catch::Approx(0.0));
        }
    }
}
