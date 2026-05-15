// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/vtkHdfMeshReader.hpp"

#include <vtkDataAssembly.h>
#include <vtkHDFReader.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSetCollection.h>

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

    SECTION("Uniform 2D mesh writes PDC with assembly " + execName)
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        NeoN::io::writeVtkHdf(mesh, "output_pdc.vtkhdf");

        vtkNew<vtkHDFReader> reader;
        reader->SetFileName("output_pdc.vtkhdf");
        reader->Update();

        auto* output = reader->GetOutput();
        auto* pdc = vtkPartitionedDataSetCollection::SafeDownCast(output);
        REQUIRE(pdc != nullptr);
        REQUIRE(pdc->GetNumberOfPartitionedDataSets() == 7);

        auto* assembly = pdc->GetDataAssembly();
        REQUIRE(assembly != nullptr);

        auto internalNodes = assembly->SelectNodes({"//internalMesh"});
        REQUIRE(internalNodes.size() == 1);

        auto boundaryNodes = assembly->SelectNodes({"//boundary"});
        REQUIRE(boundaryNodes.size() == 1);

        // Patch names in assembly must be xmin, xmax, ymin, ymax, zmin, zmax
        auto patchNodes = assembly->GetChildNodes(boundaryNodes[0]);
        REQUIRE(patchNodes.size() == 6);
        std::vector<std::string> expectedNames = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        for (std::size_t i = 0; i < 6; ++i)
        {
            REQUIRE(std::string(assembly->GetNodeName(patchNodes[i])) == expectedNames[i]);
        }

        std::filesystem::remove("output_pdc.vtkhdf");
    }
}
