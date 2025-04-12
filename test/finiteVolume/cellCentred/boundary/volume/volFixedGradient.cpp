// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("fixedGradient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        // unit cube mesh
        auto mesh = NeoN::createSingleCellMesh(exec);
        // the same as (exec, mesh.nCells(), mesh.boundaryMesh().offset())
        NeoN::DomainField<NeoN::scalar> domainField(exec, mesh);
        NeoN::fill(domainField.internalField(), 1.0);
        NeoN::fill(domainField.boundaryField().refGrad(), -1.0);
        NeoN::fill(domainField.boundaryField().refValue(), -1.0);
        NeoN::fill(domainField.boundaryField().valueFraction(), -1.0);
        NeoN::fill(domainField.boundaryField().value(), -1.0);

        SECTION("zeroGradient")
        {
            NeoN::scalar setValue {0};
            NeoN::Dictionary dict;
            dict.insert("fixedGradient", setValue);
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                    "fixedGradient", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(domainField);

            auto refValues = domainField.boundaryField().refGrad().copyToHost();

            for (auto boundaryValue : refValues.view(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainField.boundaryField().value().copyToHost();

            for (auto& boundaryValue : values.view(boundary->range()))
            {
                REQUIRE(boundaryValue == 1.0);
            }
        }

        SECTION("FixedGradient_10")
        {
            NeoN::scalar setValue {10};
            NeoN::Dictionary dict;
            dict.insert("fixedGradient", setValue);
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                    "fixedGradient", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(domainField);

            auto refValues = domainField.boundaryField().refGrad().copyToHost();

            for (auto boundaryValue : refValues.view(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainField.boundaryField().value().copyToHost();

            // deltaCoeffs is the inverse distance and has a value of 2.0
            // so the value is 1.0 + 10 / 2.0 = 6.0
            for (auto& boundaryValue : values.view(boundary->range()))
            {
                REQUIRE(boundaryValue == 6.0);
            }
        }
    }
}
