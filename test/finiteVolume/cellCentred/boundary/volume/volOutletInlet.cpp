// SPDX-FileCopyrightText: 2026 NeoN authors
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

TEST_CASE("outletInlet_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("scalar: outflow => fixedValue, inflow => zeroGradient " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        NeoN::Dictionary dict;
        dict.insert("type", std::string("outletInlet"));
        dict.insert("outletValue", NeoN::scalar(7.0));
        auto boundary =
            fvcc::VolumeBoundaryFactory<NeoN::scalar>::create("outletInlet", mesh, dict, 0);

        REQUIRE(boundary->attributes().fixesValue == false);

        // outflow (phi > 0) and phi = 0 (OpenFOAM's pos0): Dirichlet outletValue
        for (const NeoN::scalar flux : {2.0, 0.0})
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);

            auto phi = makeFlux(exec, mesh, flux);
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

        // inflow (phi < 0): zero-gradient (owner-cell value)
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);

            auto phi = makeFlux(exec, mesh, -2.0);
            fvcc::BoundaryContext ctx;
            ctx.insert("phi", phi);

            boundary->correctBoundaryCondition(field, ctx);

            auto [valuesH, valueFractionH] =
                copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

            for (auto& v : valuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(4.0));
            for (auto& f : valueFractionH.view(boundary->range()))
                REQUIRE(f == Catch::Approx(0.0));
        }

        // no context => no flux => zero-gradient
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

    SECTION("vector: outflow => fixedValue, inflow => zeroGradient " + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        const NeoN::Vec3 outletValue(1.0, 2.0, 3.0);
        const NeoN::Vec3 internal(-1.0, -1.0, -1.0);

        NeoN::Dictionary dict;
        dict.insert("type", std::string("outletInlet"));
        dict.insert("outletValue", outletValue);
        auto boundary =
            fvcc::VolumeBoundaryFactory<NeoN::Vec3>::create("outletInlet", mesh, dict, 0);

        for (const NeoN::scalar flux : {2.0, -2.0})
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), internal);

            auto phi = makeFlux(exec, mesh, flux);
            fvcc::BoundaryContext ctx;
            ctx.insert("phi", phi);

            boundary->correctBoundaryCondition(field, ctx);

            auto [valuesH, valueFractionH] =
                copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

            const NeoN::Vec3 expected = flux > 0 ? outletValue : internal;
            for (auto& v : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - valuesH.data());
                REQUIRE(valueFractionH.view()[i] == Catch::Approx(flux > 0 ? 1.0 : 0.0));
                for (auto d = 0u; d < 3; ++d)
                    REQUIRE(v[d] == Catch::Approx(expected[d]));
            }
        }
    }
}
