// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("symmetry_surface")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("TestDerivedClass" + execName)
    {
        auto mesh = NeoN::createSingleCellMesh(exec);

        // === scalar field =====================================================
        // Surface symmetry for scalars zeros the flux at the symmetry plane
        {
            auto field = NeoN::Field<NeoN::scalar>(
                exec, mesh.nInternalFaces(), mesh.boundaryMesh().offset()
            );
            NeoN::fill(field.boundaryData().refValue(), 5.0);
            NeoN::fill(field.boundaryData().value(), 5.0);

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::scalar>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto [refValuesH, valuesH] =
                copyToHosts(field.boundaryData().refValue(), field.boundaryData().value());

            for (auto& v : refValuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(0.0));
            for (auto& v : valuesH.view(boundary->range()))
                REQUIRE(v == Catch::Approx(0.0));
        }

        // === vector field =====================================================
        // Surface symmetry for Vec3 zeros the normal component of the existing boundary value
        {
            const NeoN::Vec3 inputVal(1.0, 2.0, 3.0);
            auto field =
                NeoN::Field<NeoN::Vec3>(exec, mesh.nInternalFaces(), mesh.boundaryMesh().offset());
            NeoN::fill(field.boundaryData().refValue(), NeoN::Vec3(-1.0, -1.0, -1.0));
            NeoN::fill(field.boundaryData().value(), inputVal);

            NeoN::Dictionary dict;
            auto boundary =
                NeoN::finiteVolume::cellCentred::SurfaceBoundaryFactory<NeoN::Vec3>::create(
                    "symmetry", mesh, dict, 0
                );

            boundary->correctBoundaryCondition(field);

            auto [refValuesH, valuesH, nHatH] = copyToHosts(
                field.boundaryData().refValue(),
                field.boundaryData().value(),
                mesh.boundaryMesh().faceUnitNormals()
            );

            for (auto& boundaryValueV : valuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&boundaryValueV - valuesH.data());
                const auto nV = nHatH.view()[i];
                const auto vExpected = inputVal - nV * (inputVal & nV);
                for (auto d = 0u; d < 3; ++d)
                {
                    REQUIRE(boundaryValueV[d] == Catch::Approx(vExpected[d]));
                }
            }

            for (auto& boundaryValueV : refValuesH.view(boundary->range()))
            {
                const auto i = static_cast<NeoN::localIdx>(&boundaryValueV - refValuesH.data());
                const auto nV = nHatH.view()[i];
                const auto vExpected = inputVal - nV * (inputVal & nV);
                for (auto d = 0u; d < 3; ++d)
                {
                    REQUIRE(boundaryValueV[d] == Catch::Approx(vExpected[d]));
                }
            }
        }
    }
}
