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
        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);
        NeoN::fill(field.boundaryData().refGrad(), -1.0);
        NeoN::fill(field.boundaryData().refValue(), -1.0);
        NeoN::fill(field.boundaryData().valueFraction(), -1.0);
        NeoN::fill(field.boundaryData().value(), -1.0);
        NeoN::scalar setValue {10};
        NeoN::Dictionary dict;
        dict.insert("fixedValue", setValue);
        auto boundary =
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 0
            );

        boundary->correctBoundaryCondition(field);

        auto refValues = field.boundaryData().refValue().copyToHost();

        for (auto& boundaryValue : refValues.view(boundary->range()))
        {
            REQUIRE(boundaryValue == setValue);
        }

        auto values = field.boundaryData().value().copyToHost();

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

    SECTION("PerFaceValues" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);
        auto field = NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
        NeoN::fill(field.internalVector(), 1.0);
        NeoN::fill(field.boundaryData().refGrad(), -1.0);
        NeoN::fill(field.boundaryData().refValue(), -1.0);
        NeoN::fill(field.boundaryData().valueFraction(), -1.0);
        NeoN::fill(field.boundaryData().value(), -1.0);

        const auto& offset = mesh.boundaryMesh().offset();
        const auto patchSize = static_cast<size_t>(offset[1] - offset[0]);

        // one distinct value per face, as a nonuniform OpenFOAM patch field would supply
        std::vector<NeoN::scalar> perFace(patchSize);
        for (size_t i = 0; i < patchSize; ++i)
        {
            perFace[i] = 10.0 + static_cast<NeoN::scalar>(i);
        }

        NeoN::Dictionary dict;
        dict.insert("fixedValues", perFace);
        auto boundary =
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 0
            );

        boundary->correctBoundaryCondition(field);

        auto refValues = field.boundaryData().refValue().copyToHost();
        auto values = field.boundaryData().value().copyToHost();

        size_t i = 0;
        for (auto& boundaryValue : values.view(boundary->range()))
        {
            REQUIRE(boundaryValue == perFace[i++]);
        }
        i = 0;
        for (auto& boundaryValue : refValues.view(boundary->range()))
        {
            REQUIRE(boundaryValue == perFace[i++]);
        }

        // untouched patches keep their initial value
        auto otherBoundary =
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, dict, 1
            );
        for (auto& boundaryValue : values.view(otherBoundary->range()))
        {
            REQUIRE(boundaryValue == -1.0);
        }
    }

    SECTION("RejectsMalformedDictionary" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        const auto& offset = mesh.boundaryMesh().offset();
        const auto patchSize = static_cast<size_t>(offset[1] - offset[0]);
        REQUIRE(patchSize > 0);

        // neither a uniform nor a per-face entry: nothing to apply
        NeoN::Dictionary empty;
        REQUIRE_THROWS_AS(
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, empty, 0
            ),
            NeoN::NeoNException
        );

        // a per-face list that does not match the patch
        NeoN::Dictionary wrongSize;
        wrongSize.insert("fixedValues", std::vector<NeoN::scalar>(patchSize + 1, 10.0));
        REQUIRE_THROWS_AS(
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, wrongSize, 0
            ),
            NeoN::NeoNException
        );

        // an empty per-face list is malformed too: accepting it would fall back to the
        // default-constructed uniform value and silently write zero over the patch
        NeoN::Dictionary emptyList;
        emptyList.insert("fixedValues", std::vector<NeoN::scalar> {});
        REQUIRE_THROWS_AS(
            NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                "fixedValue", mesh, emptyList, 0
            ),
            NeoN::NeoNException
        );
    }
}
