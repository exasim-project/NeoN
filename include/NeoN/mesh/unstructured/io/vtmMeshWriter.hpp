// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

/**
 * @brief Write an UnstructuredMesh to a VTM (multi-block) file for ParaView visualization.
 *
 * Writes a vtkMultiBlockDataSet containing the volume grid as block 0 ("internalMesh")
 * and each boundary patch as subsequent vtkPolyData blocks with named patches.
 *
 * @param mesh The mesh to write.
 * @param filePath Output path (must end in .vtm).
 */
void writeVtm(const UnstructuredMesh& mesh, const std::string& filePath);

} // namespace NeoN::io
