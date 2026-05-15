// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

/**
 * @brief Write an UnstructuredMesh to a VTU file for ParaView visualization.
 *
 * Writes points and cell connectivity using VTK's XML unstructured grid
 * format. Cell connectivity is recovered from the face topology stored
 * in the mesh's stencilDB.
 *
 * @param mesh The mesh to write.
 * @param filePath Output path (must end in .vtu).
 */
void writeVtu(const UnstructuredMesh& mesh, const std::string& filePath);

} // namespace NeoN::io
