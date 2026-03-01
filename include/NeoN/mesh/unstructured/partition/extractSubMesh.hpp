// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::partition
{

/// Extract the sub-mesh for partition @p partId from the global @p mesh.
///
/// @param mesh        The global unstructured mesh.
/// @param cellPart    Cell-to-part assignment (from partitionMesh).
/// @param partId      Which part to extract.
/// @returns A standalone UnstructuredMesh containing only the cells of @p partId.
///          Inter-partition faces become boundary faces under a patch named
///          "procBoundary_<partId>".
UnstructuredMesh
extractSubMesh(const UnstructuredMesh& mesh, const std::vector<int>& cellPart, int partId);

} // namespace NeoN::partition
