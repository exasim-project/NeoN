// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtmMeshWriter.hpp"

#include <vtkMultiBlockDataSet.h>
#include <vtkXMLMultiBlockDataReader.h>
#include <vtkNew.h>

#include <filesystem>

TEST_CASE("VtmMeshWriter")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Write single tet as VTM " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/singleTet.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtm(mesh, "output_singleTet.vtm"));

        REQUIRE(std::filesystem::exists("output_singleTet.vtm"));
        REQUIRE(std::filesystem::file_size("output_singleTet.vtm") > 0);

        std::filesystem::remove("output_singleTet.vtm");
        std::filesystem::remove_all("output_singleTet");
    }

    SECTION("Write cube3D as VTM " + execName)
    {
        auto mesh = NeoN::io::readCgns("meshFiles/cube3D.cgns", exec);
        REQUIRE_NOTHROW(NeoN::io::writeVtm(mesh, "output_cube3D.vtm"));

        REQUIRE(std::filesystem::exists("output_cube3D.vtm"));
        REQUIRE(std::filesystem::file_size("output_cube3D.vtm") > 0);

        std::filesystem::remove("output_cube3D.vtm");
        std::filesystem::remove_all("output_cube3D");
    }

    SECTION("Uniform 2D mesh writes 2-level multiblock " + execName)
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        NeoN::io::writeVtm(mesh, "output_mb.vtm");

        vtkNew<vtkXMLMultiBlockDataReader> reader;
        reader->SetFileName("output_mb.vtm");
        reader->Update();

        auto* mb = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
        REQUIRE(mb != nullptr);
        // 2 root blocks: internalMesh + boundary
        REQUIRE(mb->GetNumberOfBlocks() == 2);

        // Block 1 is nested boundary multiblock with 6 patches
        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);
        REQUIRE(boundary->GetNumberOfBlocks() == 6);

        std::filesystem::remove("output_mb.vtm");
        std::filesystem::remove_all("output_mb");
    }
}
