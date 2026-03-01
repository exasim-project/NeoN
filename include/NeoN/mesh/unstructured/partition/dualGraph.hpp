// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::partition
{

/// CSR cell-cell adjacency graph (METIS-compatible integer widths).
struct DualGraph
{
    std::vector<std::int32_t> xadj;   ///< row pointers, size nCells+1
    std::vector<std::int32_t> adjncy; ///< column indices (cell-cell neighbours)
    std::int32_t nCells {0};
};

/// Build the dual (cell-cell) graph of @p mesh using only internal faces.
DualGraph buildDualGraph(const UnstructuredMesh& mesh);

} // namespace NeoN::partition
