// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

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

            auto [refValuesH, valuesH, refGradH, faceCellsH, internalH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                field.boundaryData().refGrad(),
                mesh.boundaryMesh().faceOwners(),
                field.internalVector()
            );

            for (auto& boundaryValueV : refValuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&boundaryValueV - refValuesH.data());
                const auto ownerV = faceCellsH.view()[i];
                REQUIRE(boundaryValueV == Catch::Approx(internalH.view()[ownerV]));
            }

            for (auto& boundaryValueV : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&boundaryValueV - valuesH.data());
                const auto ownerV = faceCellsH.view()[i];
                REQUIRE(boundaryValueV == Catch::Approx(internalH.view()[ownerV]));
            }

            for (auto& gradValueV : refGradH.view(boundary->range()))
                REQUIRE(gradValueV == Catch::Approx(0.0));
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

            auto [refValuesH, valuesH, refGradH, nHatH, faceCellsH, internalH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                field.boundaryData().refGrad(),
                mesh.boundaryMesh().faceUnitNormals(),
                mesh.boundaryMesh().faceOwners(),
                field.internalVector()
            );

            for (auto& boundaryValueV : refValuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&boundaryValueV - refValuesH.data());
                const auto ownerV = faceCellsH.view()[i];
                const auto nV = nHatH.view()[i];
                const auto intV = internalH.view()[ownerV];
                // const auto vn = vInt & n;
                const auto vExpected = intV - nV * (intV & nV); // half-symmetry

                for (auto d = 0u; d < 3; ++d)
                {
                    REQUIRE(boundaryValueV[d] == Catch::Approx(vExpected[d]));
                }
            }

            for (auto& gradValueV : refGradH.view(boundary->range()))
            {
                for (auto d = 0u; d < 3; ++d)
                {
                    REQUIRE(gradValueV[d] == Catch::Approx(0.0));
                }
            }
        }
    }
}
