// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

// Face-flux phi whose boundary faces all carry `flux`; its sign is what selects the
// inflow/outflow branch.
fvcc::SurfaceField<NeoN::scalar>
makeFlux(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::scalar flux)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> phi(exec, "phi", mesh, bcs);
    NeoN::fill(phi.internalVector(), flux);
    NeoN::fill(phi.boundaryData().value(), flux);
    return phi;
}

// Velocity whose boundary faces all carry `u`; only its magnitude enters the dynamic head.
fvcc::VolumeField<NeoN::Vec3>
makeVelocity(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::Vec3 u)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh);
    fvcc::VolumeField<NeoN::Vec3> vel(exec, "U", mesh, bcs);
    NeoN::fill(vel.internalVector(), u);
    NeoN::fill(vel.boundaryData().value(), u);
    return vel;
}

TEST_CASE("totalPressure_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // p = p0 - 0.5*(1 - pos0(phi))*magSqr(U): p0 on outflow, p0 - 0.5*|U|^2 on inflow.
    const NeoN::scalar p0 = 5.0;
    const NeoN::Vec3 u {3.0, 4.0, 0.0}; // magSqr = 25 => dynamic head 12.5

    auto makeBoundary = [&](const NeoN::UnstructuredMesh& mesh)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("totalPressure"));
        dict.insert("p0", p0);
        return fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("totalPressure", mesh, dict, 0);
    };

    SECTION("outflow keeps p0 " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto boundary = makeBoundary(mesh);

        // fixesValue=true: the patch is Dirichlet on every face, which is what pins the
        // pressure datum here (unlike inletOutlet, which is mixed).
        REQUIRE(boundary->attributes().fixesValue == true);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);

        auto phi = makeFlux(exec, mesh, 2.0); // outflow
        auto vel = makeVelocity(exec, mesh, u);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("U", vel);

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().refGrad(),
            field.boundaryData().valueFraction()
        );

        for (auto& v : valuesH.view(boundary->range()))
            REQUIRE(v == Catch::Approx(p0));
        for (auto& v : refValuesH.view(boundary->range()))
            REQUIRE(v == Catch::Approx(p0));
        // Dirichlet: valueFraction 1, refGrad 0, on every face and in both branches.
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(1.0));
        for (auto& g : refGradH.view(boundary->range()))
            REQUIRE(g == Catch::Approx(0.0));
    }

    SECTION("inflow subtracts the dynamic head " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto boundary = makeBoundary(mesh);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);

        auto phi = makeFlux(exec, mesh, -2.0); // inflow
        auto vel = makeVelocity(exec, mesh, u);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);
        ctx.insert("U", vel);

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().refGrad(),
            field.boundaryData().valueFraction()
        );

        const NeoN::scalar expected = p0 - 0.5 * 25.0; // 5 - 12.5 = -7.5
        for (auto& v : valuesH.view(boundary->range()))
            REQUIRE(v == Catch::Approx(expected));
        for (auto& v : refValuesH.view(boundary->range()))
            REQUIRE(v == Catch::Approx(expected));
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(1.0));
        for (auto& g : refGradH.view(boundary->range()))
            REQUIRE(g == Catch::Approx(0.0));
    }

    SECTION("no context degenerates to a fixedValue at p0 " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto boundary = makeBoundary(mesh);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);

        // The documented fallback: with neither phi nor U the patch cannot know the flow
        // direction, so every face is treated as outflow.
        boundary->correctBoundaryCondition(field);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(boundary->range()))
            REQUIRE(v == Catch::Approx(p0));
        for (auto& f : valueFractionH.view(boundary->range()))
            REQUIRE(f == Catch::Approx(1.0));
    }

    SECTION("flux present but no velocity leaves p0 on inflow too " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto boundary = makeBoundary(mesh);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);

        // Inflow, but there is no velocity to build a dynamic head from.
        auto phi = makeFlux(exec, mesh, -2.0);
        fvcc::BoundaryContext ctx;
        ctx.insert("phi", phi);

        boundary->correctBoundaryCondition(field, ctx);

        auto valuesH = copyToHosts(field.boundaryData().value());
        for (auto& v : std::get<0>(valuesH).view(boundary->range()))
            REQUIRE(v == Catch::Approx(p0));
    }

    SECTION("custom phi and U names are honoured " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        NeoN::Dictionary dict;
        dict.insert("type", std::string("totalPressure"));
        dict.insert("p0", p0);
        dict.insert("phi", std::string("phiAlt"));
        dict.insert("U", std::string("Ualt"));
        auto boundary =
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("totalPressure", mesh, dict, 0);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);

        auto phi = makeFlux(exec, mesh, -2.0);
        auto vel = makeVelocity(exec, mesh, u);

        // Registered under the default names: the BC must NOT find them, so it falls back.
        {
            fvcc::BoundaryContext wrongNames;
            wrongNames.insert("phi", phi);
            wrongNames.insert("U", vel);
            boundary->correctBoundaryCondition(field, wrongNames);

            auto valuesH = copyToHosts(field.boundaryData().value());
            for (auto& v : std::get<0>(valuesH).view(boundary->range()))
                REQUIRE(v == Catch::Approx(p0));
        }

        // Registered under the configured names: the dynamic head is applied.
        {
            fvcc::BoundaryContext rightNames;
            rightNames.insert("phiAlt", phi);
            rightNames.insert("Ualt", vel);
            boundary->correctBoundaryCondition(field, rightNames);

            auto valuesH = copyToHosts(field.boundaryData().value());
            for (auto& v : std::get<0>(valuesH).view(boundary->range()))
                REQUIRE(v == Catch::Approx(p0 - 0.5 * 25.0));
        }
    }

    SECTION("omitting p0 is rejected " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        // p0 is the defining input: a missing entry is a case-setup error, not a zero datum.
        NeoN::Dictionary dict;
        dict.insert("type", std::string("totalPressure"));
        REQUIRE_THROWS(
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("totalPressure", mesh, dict, 0)
        );
    }
}
