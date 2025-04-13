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
        NeoN::Field<NeoN::scalar> domainVector(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::scalar setValue {10};
        NeoN::Dictionary dict;
        dict.insert("fixedValue", setValue);
        auto boundary =
            NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 0
            );

        boundary->correctBoundaryCondition(domainVector);

        auto refValues = domainVector.boundaryVector().refValue().copyToHost();

        for (auto& boundaryValue : refValues.view(boundary->range()))
        {
            REQUIRE(boundaryValue == setValue);
        }

        auto otherBoundary =
            NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 1
            );

        for (auto& boundaryValue : refValues.view(otherBoundary->range()))
        {
            REQUIRE(boundaryValue != setValue);
        }
    }
}
