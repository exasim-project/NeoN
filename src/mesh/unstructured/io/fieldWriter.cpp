// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/fieldWriter.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkHDFWriter.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLMultiBlockDataWriter.h>

#include <string>
#include <vector>


namespace NeoN::io
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace
{

vtkSmartPointer<vtkDoubleArray> toVtkArray(const Vector<scalar>& values, const std::string& name)
{
    auto host = values.copyToHost();
    auto view = host.view();
    localIdx n = host.size();

    vtkSmartPointer<vtkDoubleArray> arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(1);
    arr->SetNumberOfTuples(static_cast<vtkIdType>(n));
    for (localIdx i = 0; i < n; ++i)
    {
        arr->SetValue(static_cast<vtkIdType>(i), static_cast<double>(view[i]));
    }
    return arr;
}

vtkSmartPointer<vtkDoubleArray> toVtkArray(const Vector<Vec3>& values, const std::string& name)
{
    auto host = values.copyToHost();
    auto view = host.view();
    localIdx n = host.size();

    vtkSmartPointer<vtkDoubleArray> arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(3);
    arr->SetNumberOfTuples(static_cast<vtkIdType>(n));
    for (localIdx i = 0; i < n; ++i)
    {
        arr->SetTuple3(
            static_cast<vtkIdType>(i),
            static_cast<double>(view[i][0]),
            static_cast<double>(view[i][1]),
            static_cast<double>(view[i][2])
        );
    }
    return arr;
}

vtkSmartPointer<vtkDoubleArray>
sliceToVtkArray(View<const scalar> view, localIdx start, localIdx end, const std::string& name)
{
    localIdx count = end - start;
    vtkSmartPointer<vtkDoubleArray> arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(1);
    arr->SetNumberOfTuples(static_cast<vtkIdType>(count));
    for (localIdx i = 0; i < count; ++i)
    {
        arr->SetValue(static_cast<vtkIdType>(i), static_cast<double>(view[start + i]));
    }
    return arr;
}

vtkSmartPointer<vtkDoubleArray>
sliceToVtkArray(View<const Vec3> view, localIdx start, localIdx end, const std::string& name)
{
    localIdx count = end - start;
    vtkSmartPointer<vtkDoubleArray> arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(3);
    arr->SetNumberOfTuples(static_cast<vtkIdType>(count));
    for (localIdx i = 0; i < count; ++i)
    {
        const Vec3& v = view[start + i];
        arr->SetTuple3(
            static_cast<vtkIdType>(i),
            static_cast<double>(v[0]),
            static_cast<double>(v[1]),
            static_cast<double>(v[2])
        );
    }
    return arr;
}

template<typename T>
void attachField(
    vtkUnstructuredGrid* volumeGrid,
    const std::vector<vtkPolyData*>& patches,
    const fvcc::VolumeField<T>& field
)
{
    // Internal field → CellData on volumeGrid
    volumeGrid->GetCellData()->AddArray(toVtkArray(field.internalVector(), field.name));

    // Boundary field → CellData on each patch
    auto hostBndValues = field.boundaryData().value().copyToHost();
    auto view = hostBndValues.view();
    localIdx nPatches = field.boundaryData().nBoundaries();
    for (localIdx b = 0; b < nPatches; ++b)
    {
        auto [start, end] = field.boundaryData().range(b);
        patches[static_cast<std::size_t>(b)]->GetCellData()->AddArray(
            sliceToVtkArray(view, start, end, field.name)
        );
    }
}

} // anonymous namespace


FieldSet& FieldSet::add(const fvcc::VolumeField<scalar>& field)
{
    fields_.emplace_back(std::cref(field));
    return *this;
}

FieldSet& FieldSet::add(const fvcc::VolumeField<Vec3>& field)
{
    fields_.emplace_back(std::cref(field));
    return *this;
}


void writeVtm(
    const UnstructuredMesh& mesh,
    const fvcc::VolumeField<scalar>& field,
    const std::string& filePath
)
{
    auto mb = buildMultiBlockMesh(mesh);

    auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
    auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));

    std::vector<vtkPolyData*> patches;
    for (unsigned b = 0; b < boundary->GetNumberOfBlocks(); ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(boundary->GetBlock(b)));
    }

    attachField(volumeGrid, patches, field);

    vtkNew<vtkXMLMultiBlockDataWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(mb);
    writer->Write();
}

void writeVtm(
    const UnstructuredMesh& mesh, const fvcc::VolumeField<Vec3>& field, const std::string& filePath
)
{
    auto mb = buildMultiBlockMesh(mesh);

    auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
    auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));

    std::vector<vtkPolyData*> patches;
    for (unsigned b = 0; b < boundary->GetNumberOfBlocks(); ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(boundary->GetBlock(b)));
    }

    attachField(volumeGrid, patches, field);

    vtkNew<vtkXMLMultiBlockDataWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(mb);
    writer->Write();
}

void writeVtkHdf(
    const UnstructuredMesh& mesh,
    const fvcc::VolumeField<scalar>& field,
    const std::string& filePath
)
{
    auto pdc = buildPartitionedMesh(mesh);

    auto* volumeGrid =
        vtkUnstructuredGrid::SafeDownCast(pdc->GetPartitionedDataSet(0)->GetPartition(0));

    localIdx nPatches = mesh.nBoundaries();
    std::vector<vtkPolyData*> patches;
    for (localIdx b = 0; b < nPatches; ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(
            pdc->GetPartitionedDataSet(static_cast<unsigned>(b + 1))->GetPartition(0)
        ));
    }

    attachField(volumeGrid, patches, field);

    vtkNew<vtkHDFWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(pdc.Get());
    writer->Write();
}

void writeVtkHdf(
    const UnstructuredMesh& mesh, const fvcc::VolumeField<Vec3>& field, const std::string& filePath
)
{
    auto pdc = buildPartitionedMesh(mesh);

    auto* volumeGrid =
        vtkUnstructuredGrid::SafeDownCast(pdc->GetPartitionedDataSet(0)->GetPartition(0));

    localIdx nPatches = mesh.nBoundaries();
    std::vector<vtkPolyData*> patches;
    for (localIdx b = 0; b < nPatches; ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(
            pdc->GetPartitionedDataSet(static_cast<unsigned>(b + 1))->GetPartition(0)
        ));
    }

    attachField(volumeGrid, patches, field);

    vtkNew<vtkHDFWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(pdc.Get());
    writer->Write();
}

void writeVtm(const UnstructuredMesh& mesh, const FieldSet& fields, const std::string& filePath)
{
    auto mb = buildMultiBlockMesh(mesh);

    auto* volumeGrid = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
    auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));

    std::vector<vtkPolyData*> patches;
    for (unsigned b = 0; b < boundary->GetNumberOfBlocks(); ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(boundary->GetBlock(b)));
    }

    for (const auto& anyField : fields.fields_)
    {
        std::visit([&](const auto& ref) { attachField(volumeGrid, patches, ref.get()); }, anyField);
    }

    vtkNew<vtkXMLMultiBlockDataWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(mb);
    writer->Write();
}

void writeVtkHdf(const UnstructuredMesh& mesh, const FieldSet& fields, const std::string& filePath)
{
    auto pdc = buildPartitionedMesh(mesh);

    auto* volumeGrid =
        vtkUnstructuredGrid::SafeDownCast(pdc->GetPartitionedDataSet(0)->GetPartition(0));

    localIdx nPatches = mesh.nBoundaries();
    std::vector<vtkPolyData*> patches;
    for (localIdx b = 0; b < nPatches; ++b)
    {
        patches.push_back(vtkPolyData::SafeDownCast(
            pdc->GetPartitionedDataSet(static_cast<unsigned>(b + 1))->GetPartition(0)
        ));
    }

    for (const auto& anyField : fields.fields_)
    {
        std::visit([&](const auto& ref) { attachField(volumeGrid, patches, ref.get()); }, anyField);
    }

    vtkNew<vtkHDFWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(pdc.Get());
    writer->Write();
}

} // namespace NeoN::io
