// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/connectivity/types.hpp"

namespace NeoN::io
{

/// Reconstruct cell-to-node connectivity from face topology.
///
/// Collects faces per cell via faceOwner/faceNeighbour, builds unique
/// node set, determines element type from face/node counts.
/// Runs on the host (copies inputs); results are placed on exec.
CellConnectivity rebuildCellConnectivity(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    const SegmentedVector<localIdx, localIdx>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
);

/// Reconstruct per-cell info including face-node lists from face topology.
///
/// Like rebuildCellConnectivity but also stores cellFaceNodes per cell,
/// needed by node ordering functions and writers. Always host-side.
std::vector<CellInfo> rebuildCellInfo(
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    const SegmentedVector<localIdx, localIdx>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
);

} // namespace NeoN::io
