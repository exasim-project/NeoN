// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using Catch::Approx;

TEST_CASE("symmetry_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        // === scalar field =====================================================
        {
            auto field =
                NeoN::Field<NeoN::scalar>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), 4.0);
            NeoN::fill(field.boundaryData().refGrad(), -1.0);
            NeoN::fill(field.boundaryData().refValue(), -1.0);
            NeoN::fill(field.boundaryData().valueFraction(), -1.0);
            NeoN::fill(field.boundaryData().value(), -1.0);

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::scalar>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto refValues = field.boundaryData().refValue().copyToHost();
            auto values = field.boundaryData().value().copyToHost();
            auto refGrad = field.boundaryData().refGrad().copyToHost();
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

            for (auto& gradValue : refGrad.view(boundary->range()))
                REQUIRE(gradValue == Approx(0.0));
        }

        // === vector field =====================================================
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), NeoN::Vec3(1.0, -1.0, 0.5));
            NeoN::fill(field.boundaryData().refGrad(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().refValue(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().value(), NeoN::Vec3(-1.0, -1.0, -1.0));

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::Vec3>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto refValues = field.boundaryData().refValue().copyToHost();
            auto values = field.boundaryData().value().copyToHost();
            auto refGrad = field.boundaryData().refGrad().copyToHost();
            auto nHat = mesh.boundaryMesh().nf().copyToHost();
            auto faceCells = mesh.boundaryMesh().faceCells().copyToHost();
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

            for (auto& gradValue : refGrad.view(boundary->range()))
            {
                for (int d = 0; d < 3; ++d)
                    REQUIRE(gradValue[d] == Approx(0.0));
            }
        }
    }
}
