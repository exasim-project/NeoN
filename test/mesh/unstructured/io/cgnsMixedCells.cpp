// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"

#include <cmath>
#include <filesystem>

TEST_CASE("Mixed cell mesh")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Read mixed-cell mesh " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/mixedCells.cgns", exec);
        REQUIRE(mesh.nCells() > 0);
        REQUIRE(mesh.nFaces() > 0);

        // Total volume should be 1.0 (unit cube)
        auto vol = mesh.cellVolumes().copyToHost();
        NeoN::scalar sum = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            sum += vol.view()[i];
        }
        REQUIRE(sum == Catch::Approx(1.0).margin(1e-6));
    }

    SECTION("Round-trip mixed-cell mesh " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/mixedCells.cgns", exec);
        NeoN::io::writeCgns(mesh, "output_mixed.cgns");
        auto mesh2 = NeoN::io::readCgns("output_mixed.cgns", exec);

        REQUIRE(mesh2.nCells() == mesh.nCells());
        REQUIRE(mesh2.nFaces() == mesh.nFaces());

        // Volume preserved
        auto vol1 = mesh.cellVolumes().copyToHost();
        auto vol2 = mesh2.cellVolumes().copyToHost();
        NeoN::scalar sum1 = 0, sum2 = 0;
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            sum1 += vol1.view()[i];
        }
        for (NeoN::localIdx i = 0; i < mesh2.nCells(); ++i)
        {
            sum2 += vol2.view()[i];
        }
        REQUIRE(sum1 == Catch::Approx(sum2).margin(1e-6));

        std::filesystem::remove("output_mixed.cgns");
    }

    SECTION("All cell volumes positive " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/mixedCells.cgns", exec);
        auto vol = mesh.cellVolumes().copyToHost();
        for (NeoN::localIdx i = 0; i < mesh.nCells(); ++i)
        {
            REQUIRE(vol.view()[i] > 0);
        }
    }
}
