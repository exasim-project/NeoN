// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/vtmMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <vtkNew.h>
#include <vtkXMLMultiBlockDataWriter.h>

#include <string>


namespace NeoN::io
{

void writeVtm(const UnstructuredMesh& mesh, const std::string& filePath)
{
    auto mb = buildMultiBlockMesh(mesh);

    vtkNew<vtkXMLMultiBlockDataWriter> writer;
    writer->SetFileName(filePath.c_str());
    writer->SetInputData(mb.Get());
    writer->Write();
}

} // namespace NeoN::io
