// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using Catch::Approx;

TEST_CASE("symmetry_surface")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        // === scalar field =====================================================
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 5.0);
            NeoN::fill(field.boundaryData().refValue(), -1.0);
            NeoN::fill(field.boundaryData().value(), -1.0);

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::scalar>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto refValues = field.boundaryData().refValue().copyToHost();
            auto values = field.boundaryData().value().copyToHost();
            auto faceCells = mesh.boundaryMesh().faceCells().copyToHost();
            auto internal = field.internalVector().copyToHost();

            for (auto& boundaryValue : refValues.view(boundary->range()))
            {
                const auto i = &boundaryValue - refValues.data();
                const auto owner = faceCells.view()[i];
                REQUIRE(boundaryValue == Approx(internal.view()[owner]));
            }

            for (auto& boundaryValue : values.view(boundary->range()))
            {
                const auto i = &boundaryValue - values.data();
                const auto owner = faceCells.view()[i];
                REQUIRE(boundaryValue == Approx(internal.view()[owner]));
            }
        }

        // === vector field =====================================================
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), NeoN::Vec3(1.0, 2.0, 3.0));
            NeoN::fill(field.boundaryData().refValue(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().value(), NeoN::Vec3(-1.0, -1.0, -1.0));

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::Vec3>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto refValues = field.boundaryData().refValue().copyToHost();
            auto values = field.boundaryData().value().copyToHost();
            auto faceCells = mesh.boundaryMesh().faceCells().copyToHost();
            auto nHat = mesh.boundaryMesh().nf().copyToHost();
            auto internal = field.internalVector().copyToHost();

            for (auto& boundaryValue : refValues.view(boundary->range()))
            {
                const auto i = &boundaryValue - refValues.data();
                const auto owner = faceCells.view()[i];
                const auto n = nHat.view()[i];
                const auto vInt = internal.view()[owner];
                const auto vn = vInt & n;
                const auto vExpected = vInt - n * vn; // half-symmetry

                for (int d = 0; d < 3; ++d)
                    REQUIRE(boundaryValue[d] == Approx(vExpected[d]));
            }

            for (auto& boundaryValue : values.view(boundary->range()))
            {
                const auto i = &boundaryValue - values.data();
                const auto owner = faceCells.view()[i];
                const auto n = nHat.view()[i];
                const auto vInt = internal.view()[owner];
                const auto vn = vInt & n;
                const auto vExpected = vInt - n * vn;

                for (int d = 0; d < 3; ++d)
                    REQUIRE(boundaryValue[d] == Approx(vExpected[d]));
            }
        }
    }
}
