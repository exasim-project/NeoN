// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("fixedValue")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);
        NeoN::fill(field.boundaryVector().refGrad(), -1.0);
        NeoN::fill(field.boundaryVector().refValue(), -1.0);
        NeoN::fill(field.boundaryVector().valueFraction(), -1.0);
        NeoN::fill(field.boundaryVector().value(), -1.0);
        NeoN::scalar setValue {10};
        NeoN::Dictionary dict;
        dict.insert("fixedValue", setValue);
        auto boundary =
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 0
            );

        boundary->correctBoundaryCondition(field);

        auto refValues = field.boundaryVector().refValue().copyToHost();

        for (auto& boundaryValue : refValues.view(boundary->range()))
        {
            REQUIRE(boundaryValue == setValue);
        }

        auto values = field.boundaryVector().value().copyToHost();

        for (auto& boundaryValue : values.view(boundary->range()))
        {
            REQUIRE(boundaryValue == setValue);
        }

        auto otherBoundary =
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 1
            );

        for (auto& boundaryValue : refValues.view(otherBoundary->range()))
        {
            REQUIRE(boundaryValue != setValue);
        }


        for (auto& boundaryValue : values.view(otherBoundary->range()))
        {
            REQUIRE(boundaryValue != setValue);
        }
    }
}
