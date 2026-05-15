// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"


TEST_CASE("CgnsMeshReader")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single tetrahedron " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);

        REQUIRE(mesh.nCells() == 1);
        REQUIRE(mesh.nInternalFaces() == 0);
        REQUIRE(mesh.nBoundaryFaces() == 4);
        REQUIRE(mesh.nFaces() == 4);

        // Volume of tet with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1) = 1/6
        auto hostVol = mesh.cellVolumes().copyToHost();
        REQUIRE(hostVol.view()[0] == Catch::Approx(1.0 / 6.0).margin(1e-10));
    }

    SECTION("Cube 3D " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);

        REQUIRE(mesh.nCells() > 0);
        REQUIRE(mesh.nBoundaries() > 0);

        // Total volume should be 1.0
        auto hostVol = mesh.cellVolumes().copyToHost();
        NeoN::scalar totalVol = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            totalVol += hostVol.view()[i];
        }
        REQUIRE(totalVol == Catch::Approx(1.0).margin(1e-10));
    }

    SECTION("Cube 3D boundary patches " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);

        // cube3D has 6 named boundary patches
        REQUIRE(mesh.nBoundaries() == 6);

        // Total boundary faces should be consistent
        auto const& bMesh = mesh.boundaryMesh();
        auto const& offset = bMesh.offset();
        REQUIRE(offset.size() == 7); // nBoundaries + 1
        NeoN::localIdx totalBndFaces = offset.back() - offset.front();
        REQUIRE(totalBndFaces == mesh.nBoundaryFaces());
    }

    SECTION("Cube 3D face areas sum " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);

        // Sum of all boundary face area magnitudes should be 6.0
        // (surface area of unit cube)
        auto hostMagSf = mesh.boundaryMesh().magSf().copyToHost();
        NeoN::scalar totalArea = 0;
        for (NeoN::localIdx i = 0; i < mesh.nBoundaryFaces(); ++i)
        {
            totalArea += hostMagSf.view()[i];
        }
        REQUIRE(totalArea == Catch::Approx(6.0).margin(1e-8));
    }

    SECTION("Face owner/neighbour consistency " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);

        auto hostOwner = mesh.faceOwner().copyToHost();
        auto hostNeighbour = mesh.faceNeighbour().copyToHost();

        // All owners should be valid cell indices
        for (NeoN::localIdx i = 0; i < mesh.nFaces(); ++i)
        {
            REQUIRE(hostOwner.view()[i] >= 0);
            REQUIRE(hostOwner.view()[i] < mesh.nCells());
        }

        // Internal face neighbours should be valid and different from owner
        for (NeoN::localIdx i = 0; i < mesh.nInternalFaces(); ++i)
        {
            REQUIRE(hostNeighbour.view()[i] >= 0);
            REQUIRE(hostNeighbour.view()[i] < mesh.nCells());
            REQUIRE(hostOwner.view()[i] != hostNeighbour.view()[i]);
        }
    }
}
