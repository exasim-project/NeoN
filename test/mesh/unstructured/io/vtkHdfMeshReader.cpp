// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"

#include <filesystem>

TEST_CASE("VtkHdfMeshReader")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Round-trip single tet: CGNS -> VTKHDF -> read back " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        NeoN::io::writeVtkHdf(mesh, "output_singleTet.vtkhdf");

        auto mesh2 = NeoN::io::readVtkHdf("output_singleTet.vtkhdf", exec);

        REQUIRE(mesh2.nCells() == mesh.nCells());
        REQUIRE(mesh2.nFaces() == mesh.nFaces());
        REQUIRE(mesh2.nInternalFaces() == mesh.nInternalFaces());
        REQUIRE(mesh2.nBoundaryFaces() == mesh.nBoundaryFaces());

        auto hostVol = mesh2.cellVolumes().copyToHost();
        REQUIRE(hostVol.view()[0] == Catch::Approx(1.0 / 6.0).margin(1e-10));

        std::filesystem::remove("output_singleTet.vtkhdf");
    }

    SECTION("Round-trip cube3D: CGNS -> VTKHDF -> read back " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        NeoN::io::writeVtkHdf(mesh, "output_cube3D.vtkhdf");

        auto mesh2 = NeoN::io::readVtkHdf("output_cube3D.vtkhdf", exec);

        REQUIRE(mesh2.nCells() == mesh.nCells());
        REQUIRE(mesh2.nFaces() == mesh.nFaces());
        REQUIRE(mesh2.nInternalFaces() == mesh.nInternalFaces());
        REQUIRE(mesh2.nBoundaryFaces() == mesh.nBoundaryFaces());

        auto hostVol = mesh2.cellVolumes().copyToHost();
        NeoN::scalar totalVol = 0;
        for (NeoN::localIdx i = 0; i < mesh2.nCells(); ++i)
            totalVol += hostVol.view()[i];
        REQUIRE(totalVol == Catch::Approx(1.0).margin(1e-10));

        std::filesystem::remove("output_cube3D.vtkhdf");
    }

    SECTION("Face owner/neighbour consistency " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        NeoN::io::writeVtkHdf(mesh, "output_cube3D.vtkhdf");

        auto mesh2 = NeoN::io::readVtkHdf("output_cube3D.vtkhdf", exec);

        auto hostOwner = mesh2.faceOwner().copyToHost();
        auto hostNeighbour = mesh2.faceNeighbour().copyToHost();

        for (NeoN::localIdx i = 0; i < mesh2.nFaces(); ++i)
        {
            REQUIRE(hostOwner.view()[i] >= 0);
            REQUIRE(hostOwner.view()[i] < mesh2.nCells());
        }

        for (NeoN::localIdx i = 0; i < mesh2.nInternalFaces(); ++i)
        {
            REQUIRE(hostNeighbour.view()[i] >= 0);
            REQUIRE(hostNeighbour.view()[i] < mesh2.nCells());
            REQUIRE(hostOwner.view()[i] != hostNeighbour.view()[i]);
        }

        std::filesystem::remove("output_cube3D.vtkhdf");
    }
}
