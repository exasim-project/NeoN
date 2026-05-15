// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/vtkHdfMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <vtkHDFWriter.h>
#include <vtkNew.h>

#include <string>


namespace NeoN::io
{

void writeVtkHdf(const UnstructuredMesh& mesh, const std::string& filePath)
{
    auto pdc = buildPartitionedMesh(mesh);

    vtkNew<vtkHDFWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(pdc.Get());
    writer->Write();
}

} // namespace NeoN::io
