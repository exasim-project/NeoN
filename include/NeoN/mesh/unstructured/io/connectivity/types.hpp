// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/core/vector/vector.hpp"

#include <vector>

namespace NeoN::io
{

/// Cell connectivity: cell-to-node mapping and element type per cell.
struct CellConnectivity
{
    SegmentedVector<localIdx, localIdx> cellToNodes; // on exec
    Vector<int32_t> cellTypes;                       // VTK cell type IDs (10=TET, 12=HEX, etc.)
    localIdx nCells {};
};


/// Per-cell info with face-node lists for node ordering.
/// Host-side only: used during IO write path (node ordering, sequential graph traversal).
struct CellInfo
{
    std::vector<localIdx> nodeIds;                    // unique node IDs for this cell
    std::vector<std::vector<localIdx>> cellFaceNodes; // nodes of each face of this cell
    int cellType {};                                  // VTK cell type ID
};


/// Face topology: owner/neighbour, face-to-node, internal/boundary split.
struct FaceTopology
{
    Vector<localIdx> faceOwner;                    // on exec (all faces)
    Vector<localIdx> faceNeighbour;                // on exec (internal faces only)
    SegmentedVector<localIdx, localIdx> faceNodes; // on exec
    localIdx nInternalFaces {};
    localIdx nBoundaryFaces {};
};

} // namespace NeoN::io
