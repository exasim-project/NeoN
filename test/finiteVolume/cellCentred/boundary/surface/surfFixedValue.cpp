// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

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

        auto refValues = domainVector.boundaryData().refValue().copyToHost();

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

TEST_CASE("SurfaceBoundary rejects invalid input")
{
    using Boundary = NeoN::finiteVolume::cellCentred::SurfaceBoundary<NeoN::scalar>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto mesh = NeoN::createSingleCellMesh(exec);

    NeoN::Dictionary fixedValue(
        {{"type", std::string("fixedValue")}, {"fixedValue", NeoN::scalar(1)}}
    );
    NeoN::Dictionary unknown(
        {{"type", std::string("noSuchBoundary")}, {"fixedValue", NeoN::scalar(1)}}
    );
    NeoN::Dictionary noType({{"fixedValue", NeoN::scalar(1)}});

    SECTION("valid input" + execName) { REQUIRE(Boundary(mesh, fixedValue, 3).patchID() == 3); }

    SECTION("patch out of range" + execName)
    {
        // the single cell mesh has four patches
        REQUIRE_THROWS_AS(Boundary(mesh, fixedValue, 4), NeoN::NeoNException);
        REQUIRE_THROWS_AS(Boundary(mesh, fixedValue, 1000000), NeoN::NeoNException);
    }

    SECTION("unknown or missing type" + execName)
    {
        REQUIRE_THROWS_AS(Boundary(mesh, unknown, 0), NeoN::NeoNException);
        REQUIRE_THROWS_AS(Boundary(mesh, noType, 0), NeoN::NeoNException);
    }
}
