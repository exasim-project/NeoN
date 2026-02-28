// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

/**
 * @brief Write an UnstructuredMesh to a CGNS file.
 *
 * Writes the mesh points, cell connectivity, and boundary conditions
 * using the CGNS mid-level C API. Cell connectivity is recovered from
 * the face topology stored in the mesh.
 *
 * @param mesh The mesh to write.
 * @param filePath Output path (must end in .cgns).
 */
void writeCgns(const UnstructuredMesh& mesh, const std::string& filePath);

} // namespace NeoN::io
