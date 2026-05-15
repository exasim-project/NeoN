// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/fieldWriter.hpp"

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLMultiBlockDataReader.h>

#include <filesystem>

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace
{

// Build a 2x2 uniform grid mesh with 4 cells and 6 boundary patches.
NeoN::UnstructuredMesh makeMesh(const NeoN::Executor& exec)
{
    return NeoN::createUniform2DGrid(exec, 2, 2);
}

// Build a scalar VolumeField with internal value 1.0 and fixedValue BCs = 0.0.
fvcc::VolumeField<NeoN::scalar>
makeScalarField(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh)
{
    std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
    for (NeoN::localIdx p = 0; p < mesh.nBoundaries(); ++p)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 0.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, p));
    }
    NeoN::Vector<NeoN::scalar> internal(exec, mesh.nCells(), 1.0);
    fvcc::VolumeField<NeoN::scalar> phi(exec, "pressure", mesh, internal, bcs);
    phi.correctBoundaryConditions();
    return phi;
}

// Build a Vec3 VolumeField with internal value (2,3,4) and fixedValue BCs = (0,0,0).
fvcc::VolumeField<NeoN::Vec3>
makeVec3Field(const NeoN::Executor& exec, const NeoN::UnstructuredMesh& mesh)
{
    std::vector<fvcc::VolumeBoundary<NeoN::Vec3>> bcs;
    for (NeoN::localIdx p = 0; p < mesh.nBoundaries(); ++p)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", NeoN::Vec3(0.0, 0.0, 0.0));
        bcs.push_back(fvcc::VolumeBoundary<NeoN::Vec3>(mesh, dict, p));
    }
    NeoN::Vector<NeoN::Vec3> internal(exec, mesh.nCells(), NeoN::Vec3(2.0, 3.0, 4.0));
    fvcc::VolumeField<NeoN::Vec3> vel(exec, "velocity", mesh, internal, bcs);
    vel.correctBoundaryConditions();
    return vel;
}

} // anonymous namespace


TEST_CASE("fieldWriter - scalar VTM")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("scalar VolumeField is written to VTM with CellData on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto phi = makeScalarField(exec, mesh);

        const std::string path = "output_fieldWriter_scalar.vtm";
        REQUIRE_NOTHROW(NeoN::io::writeVtm(mesh, phi, path));
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::file_size(path) > 0);

        // Read back and verify CellData on volume block
        vtkNew<vtkXMLMultiBlockDataReader> reader;
        reader->SetFileName(path.c_str());
        reader->Update();

        auto* mb = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
        REQUIRE(mb != nullptr);

        auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        REQUIRE(volumeGrid != nullptr);

        auto* arr = vtkDoubleArray::SafeDownCast(volumeGrid->GetCellData()->GetArray("pressure"));
        REQUIRE(arr != nullptr);
        REQUIRE(arr->GetNumberOfTuples() == 4); // 2x2 = 4 cells

        for (vtkIdType i = 0; i < arr->GetNumberOfTuples(); ++i)
        {
            REQUIRE(arr->GetValue(i) == Catch::Approx(1.0));
        }

        // Verify all 6 patch arrays are present
        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);
        REQUIRE(boundary->GetNumberOfBlocks() == 6);

        for (unsigned b = 0; b < 6; ++b)
        {
            auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(b));
            REQUIRE(patch != nullptr);
            auto* patchArr = patch->GetCellData()->GetArray("pressure");
            REQUIRE(patchArr != nullptr);
        }

        std::filesystem::remove(path);
        // VTM also writes a companion directory
        std::filesystem::remove_all("output_fieldWriter_scalar");
    }
}

TEST_CASE("fieldWriter - scalar VTKHDF")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("scalar VolumeField is written to VTKHDF on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto phi = makeScalarField(exec, mesh);

        const std::string path = "output_fieldWriter_scalar.vtkhdf";
        REQUIRE_NOTHROW(NeoN::io::writeVtkHdf(mesh, phi, path));
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::file_size(path) > 0);

        std::filesystem::remove(path);
    }
}

TEST_CASE("fieldWriter - Vec3 VTM")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Vec3 VolumeField is written to VTM with 3-component CellData on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto vel = makeVec3Field(exec, mesh);

        const std::string path = "output_fieldWriter_vec3.vtm";
        REQUIRE_NOTHROW(NeoN::io::writeVtm(mesh, vel, path));
        REQUIRE(std::filesystem::exists(path));

        vtkNew<vtkXMLMultiBlockDataReader> reader;
        reader->SetFileName(path.c_str());
        reader->Update();

        auto* mb = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
        REQUIRE(mb != nullptr);

        auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        REQUIRE(volumeGrid != nullptr);

        auto* arr = vtkDoubleArray::SafeDownCast(volumeGrid->GetCellData()->GetArray("velocity"));
        REQUIRE(arr != nullptr);
        REQUIRE(arr->GetNumberOfComponents() == 3);
        REQUIRE(arr->GetNumberOfTuples() == 4);

        std::filesystem::remove(path);
        std::filesystem::remove_all("output_fieldWriter_vec3");
    }
}

TEST_CASE("fieldWriter - Vec3 VTKHDF")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Vec3 VolumeField is written to VTKHDF on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto vel = makeVec3Field(exec, mesh);

        const std::string path = "output_fieldWriter_vec3.vtkhdf";
        REQUIRE_NOTHROW(NeoN::io::writeVtkHdf(mesh, vel, path));
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::file_size(path) > 0);

        std::filesystem::remove(path);
    }
}

TEST_CASE("fieldWriter - multi-field VTM")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("FieldSet with scalar + Vec3 written to VTM on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto phi = makeScalarField(exec, mesh);
        auto vel = makeVec3Field(exec, mesh);

        NeoN::io::FieldSet fs;
        fs.add(phi).add(vel);

        const std::string path = "output_multi_field.vtm";
        REQUIRE_NOTHROW(NeoN::io::writeVtm(mesh, fs, path));
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::file_size(path) > 0);

        // Read back and verify both arrays present on volume block
        vtkNew<vtkXMLMultiBlockDataReader> reader;
        reader->SetFileName(path.c_str());
        reader->Update();

        auto* mb = vtkMultiBlockDataSet::SafeDownCast(reader->GetOutput());
        REQUIRE(mb != nullptr);

        auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        REQUIRE(volumeGrid != nullptr);

        auto* pressureArr =
            vtkDoubleArray::SafeDownCast(volumeGrid->GetCellData()->GetArray("pressure"));
        REQUIRE(pressureArr != nullptr);
        REQUIRE(pressureArr->GetNumberOfTuples() == 4);

        auto* velocityArr =
            vtkDoubleArray::SafeDownCast(volumeGrid->GetCellData()->GetArray("velocity"));
        REQUIRE(velocityArr != nullptr);
        REQUIRE(velocityArr->GetNumberOfComponents() == 3);
        REQUIRE(velocityArr->GetNumberOfTuples() == 4);

        std::filesystem::remove(path);
        std::filesystem::remove_all("output_multi_field");
    }
}

TEST_CASE("fieldWriter - multi-field VTKHDF")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("FieldSet with scalar + Vec3 written to VTKHDF on: " + execName)
    {
        auto mesh = makeMesh(exec);
        auto phi = makeScalarField(exec, mesh);
        auto vel = makeVec3Field(exec, mesh);

        NeoN::io::FieldSet fs;
        fs.add(phi).add(vel);

        const std::string path = "output_multi_field.vtkhdf";
        REQUIRE_NOTHROW(NeoN::io::writeVtkHdf(mesh, fs, path));
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::file_size(path) > 0);

        std::filesystem::remove(path);
    }
}
