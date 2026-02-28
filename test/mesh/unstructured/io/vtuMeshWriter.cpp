// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtuMeshWriter.hpp"

#include <filesystem>
#include <fstream>

TEST_CASE("VtuMeshWriter")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Write single tet as VTU " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtu(mesh, "output_singleTet.vtu"));

        std::ifstream f("output_singleTet.vtu");
        REQUIRE(f.good());
        f.seekg(0, std::ios::end);
        REQUIRE(f.tellg() > 0);

        std::filesystem::remove("output_singleTet.vtu");
    }

    SECTION("Write cube3D as VTU " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtu(mesh, "output_cube3D.vtu"));

        std::ifstream f("output_cube3D.vtu");
        REQUIRE(f.good());
        f.seekg(0, std::ios::end);
        REQUIRE(f.tellg() > 0);

        std::filesystem::remove("output_cube3D.vtu");
    }
}
