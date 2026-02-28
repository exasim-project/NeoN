// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"

#include <filesystem>

TEST_CASE("VtkHdfMeshWriter")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Write single tet as VTKHDF " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtkHdf(mesh, "output_singleTet.vtkhdf"));

        REQUIRE(std::filesystem::exists("output_singleTet.vtkhdf"));
        REQUIRE(std::filesystem::file_size("output_singleTet.vtkhdf") > 0);

        std::filesystem::remove("output_singleTet.vtkhdf");
    }

    SECTION("Write cube3D as VTKHDF " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtkHdf(mesh, "output_cube3D.vtkhdf"));

        REQUIRE(std::filesystem::exists("output_cube3D.vtkhdf"));
        REQUIRE(std::filesystem::file_size("output_cube3D.vtkhdf") > 0);

        std::filesystem::remove("output_cube3D.vtkhdf");
    }
}
