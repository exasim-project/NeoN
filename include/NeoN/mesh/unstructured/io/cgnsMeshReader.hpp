// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

/**
 * @brief Read a CGNS file and return an UnstructuredMesh on the given executor.
 *
 * Uses VTK's CGNS reader internally to parse the file, then extracts points,
 * cell connectivity, boundary conditions, and computes face topology and
 * geometric quantities.
 *
 * @param filePath Path to the .cgns file.
 * @param exec The executor to use for the mesh data.
 * @return The constructed UnstructuredMesh.
 */
UnstructuredMesh readCgns(const std::string& filePath, const Executor& exec);

} // namespace NeoN::io
