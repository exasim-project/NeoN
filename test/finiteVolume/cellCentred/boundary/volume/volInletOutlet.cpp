// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

// Build a face-flux phi (surface scalar field) whose boundary faces all carry `flux`.
fvcc::SurfaceField<NeoN::scalar>
makeFlux(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh, NeoN::scalar flux)
{
    auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh);
    fvcc::SurfaceField<NeoN::scalar> phi(exec, "phi", mesh, bcs);
    NeoN::fill(phi.internalVector(), flux);
    NeoN::fill(phi.boundaryData().value(), flux);
    return phi;
}

TEST_CASE("inletOutlet_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("scalar: inflow => fixedValue, outflow => zeroGradient " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        NeoN::Dictionary dict;
        dict.insert("type", std::string("inletOutlet"));
        dict.insert("inletValue", NeoN::scalar(7.0));
        auto boundary =
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("inletOutlet", mesh, dict, 0);

        // fixesValue=true: GaussGreenGrad uses value() for snGrad, giving the correct
        // Dirichlet gradient on inflow faces and zero gradient on outflow faces.
        REQUIRE(boundary->attributes().fixesValue == true);

        // === inflow (phi < 0): Dirichlet inletValue ============================
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);

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
                REQUIRE(v == Catch::Approx(7.0));
            for (auto& v : refValuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(7.0));
            for (auto& f : valueFractionH.view(boundary->range()))
                REQUIRE(f == Catch::Approx(1.0));
            for (auto& g : refGradH.view(boundary->range()))
                REQUIRE(g == Catch::Approx(0.0));
        }

        // === outflow (phi >= 0): zero-gradient (owner-cell value) ==============
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);

            auto phi = makeFlux(exec, mesh, 2.0);
            fvcc::BoundaryContext ctx;
            ctx.insert("phi", phi);

            boundary->correctBoundaryCondition(field, ctx);

            auto [valuesH, refGradH, valueFractionH] = copyToHosts(
                field.boundaryData().value(),
                field.boundaryData().refGrad(),
                field.boundaryData().valueFraction()
            );

            for (auto& v : valuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(4.0));
            for (auto& f : valueFractionH.view(boundary->range()))
                REQUIRE(f == Catch::Approx(0.0));
            for (auto& g : refGradH.view(boundary->range()))
                REQUIRE(g == Catch::Approx(0.0));
        }

        // === no context => no flux => zero-gradient ============================
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);

            boundary->correctBoundaryCondition(field);

            auto [valuesH, valueFractionH] =
                copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

            for (auto& v : valuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(4.0));
            for (auto& f : valueFractionH.view(boundary->range()))
                REQUIRE(f == Catch::Approx(0.0));
        }
    }

    SECTION("vector: inflow => fixedValue, outflow => zeroGradient " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        const NeoN::Vec3 inletValue(1.0, 2.0, 3.0);

        NeoN::Dictionary dict;
        dict.insert("type", std::string("inletOutlet"));
        dict.insert("inletValue", inletValue);
        auto boundary =
            fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create("inletOutlet", mesh, dict, 0);

        // inflow
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), NeoN::Vec3(-1.0, -1.0, -1.0));

            auto phi = makeFlux(exec, mesh, -2.0);
            fvcc::BoundaryContext ctx;
            ctx.insert("phi", phi);

            boundary->correctBoundaryCondition(field, ctx);

            auto [valuesH, valueFractionH] =
                copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

            for (auto& v : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - valuesH.data());
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(1.0));
                for (auto d = 0u; d < 3; ++d)
                    REQUIRE(v[d] == Catch::Approx(inletValue[d]));
            }
        }

        // outflow
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            const NeoN::Vec3 internal(-1.0, -1.0, -1.0);
            NeoN::fill(field.internalVector(), internal);

            auto phi = makeFlux(exec, mesh, 2.0);
            fvcc::BoundaryContext ctx;
            ctx.insert("phi", phi);

            boundary->correctBoundaryCondition(field, ctx);

            auto [valuesH, valueFractionH] =
                copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

            for (auto& v : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - valuesH.data());
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(0.0));
                for (auto d = 0u; d < 3; ++d)
                    REQUIRE(v[d] == Catch::Approx(internal[d]));
            }
        }
    }
}
