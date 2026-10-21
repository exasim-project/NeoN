// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("sineFixedValue_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Single-cell mesh (createSingleCellMesh); patch 0 is where SineFixedValue is attached below.
    auto mesh = NeoN::createSingleCellMesh(exec);
    const std::pair<NeoN::localIdx, NeoN::localIdx> patch0Range {0, 1};
    const NeoN::scalar ownerValue = 4.0;

    SECTION("default amplitude (1) and t0 (0), sin(pi/2) " + execName)
    {
        // frequency=3000, scale=50, level=101325, time=1/12000 -> 2*pi*frequency*time = pi/2,
        // sin(pi/2)=1 -> value = 1*50*1 + 101325 = 101375.
        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("sineFixedValue"));
        patch0Dict.insert("frequency", NeoN::scalar(3000.0));
        patch0Dict.insert("scale", NeoN::scalar(50.0));
        patch0Dict.insert("level", NeoN::scalar(101325.0));

        auto boundary = fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
            "sineFixedValue", mesh, patch0Dict, 0
        );
        REQUIRE(boundary->attributes().fixesValue == true);

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        fvcc::BoundaryContext ctx;
        ctx.insertValue("time", NeoN::scalar(1.0 / 12000.0));

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, refValuesH, refGradH, valueFractionH] = copyToHosts(
            field.boundaryData().value(),
            field.boundaryData().refValue(),
            field.boundaryData().refGrad(),
            field.boundaryData().valueFraction()
        );

        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(101375.0));
        for (auto& v : refValuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(101375.0));
        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(1.0));
        for (auto& g : refGradH.view(patch0Range))
            REQUIRE(g == Catch::Approx(0.0));
    }

    SECTION("explicit amplitude and t0 offset " + execName)
    {
        // amplitude=2, frequency=3000, scale=50, level=101325, t0=1/12000,
        // time = t0 + 1/12000 -> 2*pi*frequency*(time-t0) = pi/2, sin=1
        // -> value = 2*50*1 + 101325 = 101425.
        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("sineFixedValue"));
        patch0Dict.insert("amplitude", NeoN::scalar(2.0));
        patch0Dict.insert("frequency", NeoN::scalar(3000.0));
        patch0Dict.insert("scale", NeoN::scalar(50.0));
        patch0Dict.insert("level", NeoN::scalar(101325.0));
        patch0Dict.insert("t0", NeoN::scalar(1.0 / 12000.0));

        auto boundary = fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
            "sineFixedValue", mesh, patch0Dict, 0
        );

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        fvcc::BoundaryContext ctx;
        ctx.insertValue("time", NeoN::scalar(2.0 / 12000.0));

        boundary->correctBoundaryCondition(field, ctx);

        auto valuesH = copyToHosts(field.boundaryData().value());
        for (auto& v : std::get<0>(valuesH).view(patch0Range))
            REQUIRE(v == Catch::Approx(101425.0));
    }

    SECTION("missing 'time' in context => degenerates to zero-gradient " + execName)
    {
        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("sineFixedValue"));
        patch0Dict.insert("frequency", NeoN::scalar(3000.0));
        patch0Dict.insert("scale", NeoN::scalar(50.0));
        patch0Dict.insert("level", NeoN::scalar(101325.0));

        auto boundary = fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
            "sineFixedValue", mesh, patch0Dict, 0
        );

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        fvcc::BoundaryContext ctx; // deliberately empty, no "time"

        boundary->correctBoundaryCondition(field, ctx);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(ownerValue));
        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.0));
    }

    SECTION("no context at all => degenerates to zero-gradient " + execName)
    {
        NeoN::Dictionary patch0Dict;
        patch0Dict.insert("type", std::string("sineFixedValue"));
        patch0Dict.insert("frequency", NeoN::scalar(3000.0));
        patch0Dict.insert("scale", NeoN::scalar(50.0));
        patch0Dict.insert("level", NeoN::scalar(101325.0));

        auto boundary = fvcc::VolumeBoundaryFactory<NeoN::scalar>::create(
            "sineFixedValue", mesh, patch0Dict, 0
        );

        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), ownerValue);

        boundary->correctBoundaryCondition(field);

        auto [valuesH, valueFractionH] =
            copyToHosts(field.boundaryData().value(), field.boundaryData().valueFraction());

        for (auto& v : valuesH.view(patch0Range))
            REQUIRE(v == Catch::Approx(ownerValue));
        for (auto& f : valueFractionH.view(patch0Range))
            REQUIRE(f == Catch::Approx(0.0));
    }
}
