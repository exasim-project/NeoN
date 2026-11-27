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
 * @brief Read an UnstructuredMesh from a VTM (VTK multiblock) file.
 *
 * The VTM file must follow the layout produced by NeoN's VTM writer:
 *   - Block 0 ("internalMesh"): vtkUnstructuredGrid with all volume cells.
 *   - Block 1 ("boundary"): nested vtkMultiBlockDataSet whose sub-blocks
 *     are named vtkPolyData patches.
 *
 * Connectivity and geometry are reconstructed from the volume cells.
 * Boundary faces are matched to their originating patch by comparing
 * sorted face-node sets, so patch names and per-patch offsets are preserved.
 *
 * Patch names and face-to-node connectivity are stored in the mesh stencilDB
 * under the keys `io::patchNames` and `io::faceNodes` respectively, allowing
 * a lossless write-back via the VTM writer.
 *
 * @param filePath Path to the .vtm index file.
 * @param exec     Target executor for the returned mesh data.
 * @return         A fully initialised UnstructuredMesh on @p exec.
 * @throws std::runtime_error if the file cannot be opened or the expected
 *         block structure is absent.
 */
UnstructuredMesh readVtm(const std::string& filePath, const Executor& exec);

} // namespace NeoN::io
