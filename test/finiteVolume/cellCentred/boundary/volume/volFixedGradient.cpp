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
        NeoN::Field<NeoN::scalar> domainVector(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(domainVector.internalVector(), 1.0);
        NeoN::fill(domainVector.boundaryVector().refGrad(), -1.0);
        NeoN::fill(domainVector.boundaryVector().refValue(), -1.0);
        NeoN::fill(domainVector.boundaryVector().valueFraction(), -1.0);
        NeoN::fill(domainVector.boundaryVector().value(), -1.0);

        SECTION("zeroGradient")
        {
            NeoN::scalar setValue {0};
            NeoN::Dictionary dict;
            dict.insert("fixedGradient", setValue);
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                    "fixedGradient", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(domainVector);

            auto refValues = domainVector.boundaryVector().refGrad().copyToHost();

            for (auto boundaryValue : refValues.view(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainVector.boundaryVector().value().copyToHost();

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

            boundary->correctBoundaryCondition(domainVector);

            auto refValues = domainVector.boundaryVector().refGrad().copyToHost();

            for (auto boundaryValue : refValues.view(boundary->range()))
            {
                REQUIRE(boundaryValue == setValue);
            }

            auto values = domainVector.boundaryVector().value().copyToHost();

            // deltaCoeffs is the inverse distance and has a value of 2.0
            // so the value is 1.0 + 10 / 2.0 = 6.0
            for (auto& boundaryValue : values.view(boundary->range()))
            {
                REQUIRE(boundaryValue == 6.0);
            }
        }
    }
}
