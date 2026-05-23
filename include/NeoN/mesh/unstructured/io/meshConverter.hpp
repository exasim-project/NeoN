// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"
#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <string_view>
#include <vector>

class vtkMultiBlockDataSet;

namespace NeoN::io
{

/// StencilDB key for face-node connectivity (SegmentedVector<localIdx,localIdx>).
constexpr std::string_view stencilFaceNodes = "io::faceNodes";

/// StencilDB key for patch names (std::vector<std::string>).
constexpr std::string_view stencilPatchNames = "io::patchNames";

/**
 * @brief Build BoundaryMesh from geometry arrays.
 *
 * Extracts boundary-face quantities (centers, normals, deltas, weights, etc.)
 * from the global face arrays and assembles them into a BoundaryMesh on
 * the target executor.
 *
 * @param exec           Target executor.
 * @param faceOwner      Owner cell for every face (internal + boundary).
 * @param faceCenters    Face center vectors for every face.
 * @param cellCenters    Cell center vectors.
 * @param faceAreas      Face area vectors (outward-pointing).
 * @param magFaceAreas   Face area magnitudes.
 * @param nInternalFaces Number of internal faces (boundary faces follow).
 * @param nBoundaryFaces Number of boundary faces.
 * @param patchOffsets   Patch offset array (size = nPatches+1).
 */
BoundaryMesh buildBoundaryMesh(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<Vec3>& faceCenters,
    const Vector<Vec3>& cellCenters,
    const Vector<Vec3>& faceAreas,
    const Vector<scalar>& magFaceAreas,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    const std::vector<localIdx>& patchOffsets
);

/**
 * @brief Extract patch names from a VTK multiblock boundary sub-block.
 *
 * Returns the NAME metadata of each block in @p boundary.
 * If a block has no name, "patch_N" is used as fallback.
 */
std::vector<std::string> multiBlockPatchNames(vtkMultiBlockDataSet* boundary);

} // namespace NeoN::io
