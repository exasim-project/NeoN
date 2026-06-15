// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("slip_volume")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        // === scalar field: zero-gradient =======================================
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
                    "slip", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto [refValuesH, valuesH, refGradH, faceCellsH, internalH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                field.boundaryData().refGrad(),
                mesh.boundaryMesh().faceOwners(),
                field.internalVector()
            );

            for (auto& v : refValuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - refValuesH.data());
                REQUIRE(v == Catch::Approx(internalH.view()[faceCellsH.view()[i]]));
            }

            for (auto& v : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - valuesH.data());
                REQUIRE(v == Catch::Approx(internalH.view()[faceCellsH.view()[i]]));
            }

            for (auto& g : refGradH.view(boundary->range()))
                REQUIRE(g == Catch::Approx(0.0));
        }

        // === vector field: tangential projection ================================
        {
            auto field = NeoN::Field<NeoN::Vec3>(exec, mesh.nCells(), mesh.boundaryMesh().offset());
            NeoN::fill(field.internalVector(), NeoN::Vec3(1.0, -1.0, 0.5));
            NeoN::fill(field.boundaryData().refGrad(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().refValue(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().value(), NeoN::Vec3(-1.0, -1.0, -1.0));

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::VolumeBoundaryFactory<NeoN::Vec3>::create(
                    "slip", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto [refValuesH, valuesH, refGradH, nHatH, faceCellsH, internalH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                field.boundaryData().refGrad(),
                mesh.boundaryMesh().faceUnitNormals(),
                mesh.boundaryMesh().faceOwners(),
                field.internalVector()
            );

            for (auto& v : refValuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&v - refValuesH.data());
                const auto intV = internalH.view()[faceCellsH.view()[i]];
                const auto n = nHatH.view()[i];
                const auto vExpected = intV - n * (intV & n);

                for (auto d = 0u; d < 3; ++d)
                    REQUIRE(v[d] == Catch::Approx(vExpected[d]));
            }

            for (auto& g : refGradH.view(boundary->range()))
            {
                for (auto d = 0u; d < 3; ++d)
                    REQUIRE(g[d] == Catch::Approx(0.0));
            }
        }
    }
}
