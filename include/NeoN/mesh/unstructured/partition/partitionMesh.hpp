// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::partition
{

/// Partition @p mesh into @p nParts parts using METIS (Kway).
/// @returns A vector of size nCells where entry i holds the part id (0-based) for cell i.
/// @note When nParts==1 the trivial all-zero result is returned without invoking METIS.
std::vector<int> partitionMesh(const UnstructuredMesh& mesh, int nParts);

} // namespace NeoN::partition
