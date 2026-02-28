// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"

TEST_CASE("Geometric quantities")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single tet cell centre " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        auto centres = mesh.cellCentres().copyToHost();
        // Centroid of tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) = (0.25, 0.25, 0.25)
        REQUIRE(centres.view()[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(centres.view()[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(centres.view()[0][2] == Catch::Approx(0.25).margin(1e-10));
    }

    SECTION("Cube volume-weighted centroid " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        auto centres = mesh.cellCentres().copyToHost();
        auto volumes = mesh.cellVolumes().copyToHost();

        NeoN::Vec3 weighted {0, 0, 0};
        NeoN::scalar totalVol = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            NeoN::scalar v = volumes.view()[i];
            weighted = weighted + centres.view()[i] * v;
            totalVol += v;
        }
        weighted = weighted * (1.0 / totalVol);

        REQUIRE(weighted[0] == Catch::Approx(0.5).margin(1e-6));
        REQUIRE(weighted[1] == Catch::Approx(0.5).margin(1e-6));
        REQUIRE(weighted[2] == Catch::Approx(0.5).margin(1e-6));
    }

    SECTION("Face area magnitudes sum correctly " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        auto magSf = mesh.magFaceAreas().copyToHost();

        // Boundary face areas should sum to 6.0 (surface of unit cube)
        NeoN::scalar bndAreaSum = 0;
        for (NeoN::localIdx i = mesh.nInternalFaces(); i < mesh.nFaces(); ++i)
        {
            bndAreaSum += magSf.view()[i];
        }
        REQUIRE(bndAreaSum == Catch::Approx(6.0).margin(1e-6));
    }

    SECTION("All cell volumes positive " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        auto volumes = mesh.cellVolumes().copyToHost();

        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            REQUIRE(volumes.view()[i] > 0);
        }
    }

    SECTION("Face centres within unit cube " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        auto faceCentres = mesh.faceCentres().copyToHost();

        for (NeoN::localIdx i = 0; i < mesh.nFaces(); ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                REQUIRE(faceCentres.view()[i][d] >= -1e-10);
                REQUIRE(faceCentres.view()[i][d] <= 1.0 + 1e-10);
            }
        }
    }
}
