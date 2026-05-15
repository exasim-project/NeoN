// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"

#include <filesystem>

TEST_CASE("CgnsMeshWriter")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Write and re-read single tet " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_singleTet.cgns");

        auto mesh2 = NeoN::io::readCgns("output_singleTet.cgns", exec);
        REQUIRE(mesh2.nCells() == mesh.nCells());
        REQUIRE(mesh2.nFaces() == mesh.nFaces());
        REQUIRE(mesh2.nBoundaryFaces() == mesh.nBoundaryFaces());
        REQUIRE(mesh2.nInternalFaces() == mesh.nInternalFaces());

        // Verify volume preserved
        auto vol1 = mesh.cellVolumes().copyToHost();
        auto vol2 = mesh2.cellVolumes().copyToHost();
        NeoN::scalar sum1 = 0, sum2 = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            sum1 += vol1.view()[i];
            sum2 += vol2.view()[i];
        }
        REQUIRE(sum1 == Catch::Approx(sum2).margin(1e-10));

        std::filesystem::remove("output_singleTet.cgns");
    }

    SECTION("Write and re-read cube 3D " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_cube3D.cgns");

        auto mesh2 = NeoN::io::readCgns("output_cube3D.cgns", exec);
        REQUIRE(mesh2.nCells() == mesh.nCells());
        REQUIRE(mesh2.nFaces() == mesh.nFaces());
        REQUIRE(mesh2.nBoundaryFaces() == mesh.nBoundaryFaces());

        // Total volume preserved
        auto vol1 = mesh.cellVolumes().copyToHost();
        auto vol2 = mesh2.cellVolumes().copyToHost();
        NeoN::scalar sum1 = 0, sum2 = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            sum1 += vol1.view()[i];
            sum2 += vol2.view()[i];
        }
        REQUIRE(sum1 == Catch::Approx(sum2).margin(1e-10));

        std::filesystem::remove("output_cube3D.cgns");
    }

    SECTION("Boundary patch count preserved " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_cube3D_bc.cgns");

        auto mesh2 = NeoN::io::readCgns("output_cube3D_bc.cgns", exec);
        REQUIRE(mesh2.nBoundaries() == mesh.nBoundaries());

        // Offset arrays must match
        auto const& off1 = mesh.boundaryMesh().offset();
        auto const& off2 = mesh2.boundaryMesh().offset();
        REQUIRE(off1.size() == off2.size());
        for (std::size_t i = 0; i < off1.size(); ++i)
        {
            REQUIRE(off1[i] == off2[i]);
        }

        std::filesystem::remove("output_cube3D_bc.cgns");
    }

    SECTION("Points preserved " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_singleTet_pts.cgns");

        auto mesh2 = NeoN::io::readCgns("output_singleTet_pts.cgns", exec);

        auto pts1 = mesh.points().copyToHost();
        auto pts2 = mesh2.points().copyToHost();
        REQUIRE(pts1.size() == pts2.size());

        // Sort points for comparison (ordering may differ)
        std::vector<std::array<NeoN::scalar, 3>> sorted1(static_cast<std::size_t>(pts1.size()));
        std::vector<std::array<NeoN::scalar, 3>> sorted2(static_cast<std::size_t>(pts2.size()));
        for (NeoN::localIdx i = 0; i < pts1.size(); ++i)
        {
            auto si = static_cast<std::size_t>(i);
            sorted1[si] = {pts1.view()[i][0], pts1.view()[i][1], pts1.view()[i][2]};
            sorted2[si] = {pts2.view()[i][0], pts2.view()[i][1], pts2.view()[i][2]};
        }
        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        for (std::size_t i = 0; i < sorted1.size(); ++i)
        {
            REQUIRE(sorted1[i][0] == Catch::Approx(sorted2[i][0]).margin(1e-12));
            REQUIRE(sorted1[i][1] == Catch::Approx(sorted2[i][1]).margin(1e-12));
            REQUIRE(sorted1[i][2] == Catch::Approx(sorted2[i][2]).margin(1e-12));
        }

        std::filesystem::remove("output_singleTet_pts.cgns");
    }

    SECTION("Uniform 2D mesh roundtrip preserves 6 boundaries " + execName)
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        REQUIRE(mesh.nBoundaries() == 6);

        NeoN::io::writeCgns(mesh, "output_uniform2d.cgns");
        auto mesh2 = NeoN::io::readCgns("output_uniform2d.cgns", exec);

        REQUIRE(mesh2.nBoundaries() == 6);
        REQUIRE(mesh2.nCells() == 4);
        REQUIRE(mesh2.nBoundaryFaces() == mesh.nBoundaryFaces());

        std::filesystem::remove("output_uniform2d.cgns");
    }
}
