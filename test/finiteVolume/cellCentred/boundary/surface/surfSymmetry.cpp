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

            auto [refValuesH, valuesH, faceCellsH, internalH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                mesh.boundaryMesh().faceCells(),
                field.internalVector()
            );

            for (auto& boundaryValueV : refValuesH.view(boundary->range()))
            {
                const auto i = &boundaryValueV - refValuesH.data();
                const auto ownerV = faceCellsH.view()[i];
                REQUIRE(boundaryValueV == Approx(internalH.view()[ownerV]));
            }

            for (auto& boundaryValueV : valuesH.view(boundary->range()))
            {
                const auto i = &boundaryValueV - valuesH.data();
                const auto ownerV = faceCellsH.view()[i];
                REQUIRE(boundaryValueV == Approx(internalH.view()[ownerV]));
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

            auto [refValuesH, valuesH, faceCellsH, internalH, nHatH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                mesh.boundaryMesh().faceCells(),
                field.internalVector(),
                mesh.boundaryMesh().nf()
            );

            for (auto& boundaryValueV : refValuesH.view(boundary->range()))
            {
                const auto i = &boundaryValueV - refValuesH.data();
                const auto ownerV = faceCellsH.view()[i];
                const auto nV = nHatH.view()[i];
                const auto intV = internalH.view()[ownerV];
                // const auto vn = vInt & n;
                const auto vExpected = intV - nV * (intV & nV); // half-symmetry

                for (int d = 0; d < 3; ++d)
                    REQUIRE(boundaryValueV[d] == Approx(vExpected[d]));
            }

            for (auto& boundaryValueV : valuesH.view(boundary->range()))
            {
                const auto i = &boundaryValueV - valuesH.data();
                const auto ownerV = faceCellsH.view()[i];
                const auto nV = nHatH.view()[i];
                const auto intV = internalH.view()[ownerV];
                // const auto vn = vInt & n;
                const auto vExpected = intV - nV * (intV & nV);

                for (int d = 0; d < 3; ++d)
                    REQUIRE(boundaryValueV[d] == Approx(vExpected[d]));
            }
        }
    }
}
