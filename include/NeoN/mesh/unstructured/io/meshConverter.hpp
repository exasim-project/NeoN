// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"

#include <vector>

class vtkUnstructuredGrid;

namespace NeoN::io
{

/// Cell connectivity: cell-to-node mapping and element type per cell.
struct CellConnectivity
{
    std::vector<std::vector<localIdx>> cellToNodes;
    std::vector<int> cellTypes; // VTK cell type IDs (10=TET, 12=HEX, etc.)
    localIdx nCells {};
};


/// Per-cell info with face-node lists for node ordering.
struct CellInfo
{
    std::vector<localIdx> nodeIds;                    // unique node IDs for this cell
    std::vector<std::vector<localIdx>> cellFaceNodes; // nodes of each face of this cell
    int cellType {};                                  // VTK cell type ID
};


/// Face topology: owner/neighbour, face-to-node, internal/boundary split.
struct FaceTopology
{
    std::vector<localIdx> faceOwner;
    std::vector<localIdx> faceNeighbour;
    std::vector<std::vector<localIdx>> faceNodes;
    localIdx nInternalFaces {};
    localIdx nBoundaryFaces {};
};


/// Geometric quantities derived from mesh topology and point coordinates.
struct MeshGeometry
{
    std::vector<scalar> cellVolumes;
    std::vector<Vec3> cellCentres;
    std::vector<Vec3> faceAreas;
    std::vector<Vec3> faceCentres;
    std::vector<scalar> magFaceAreas;
};


/// Build face topology from cell-to-node connectivity.
///
/// Each cell's faces are identified using element-type face templates,
/// then deduplicated using canonical (sorted) face keys. Shared faces
/// become internal faces; unshared faces become boundary faces.
FaceTopology buildFaceTopology(const CellConnectivity& connectivity);

/// Compute all geometric quantities from points and face topology.
///
/// Face areas via triangulation from centroid, cell centres as average
/// of touching face centres, cell volumes via tetrahedral decomposition.
MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells);

/// Reconstruct cell-to-node connectivity from face topology.
///
/// Collects faces per cell via faceOwner/faceNeighbour, builds unique
/// node set, determines element type from face/node counts.
CellConnectivity rebuildCellConnectivity(
    const std::vector<label>& faceOwner,
    const std::vector<label>& faceNeighbour,
    const std::vector<std::vector<localIdx>>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
);

/// Reconstruct per-cell info including face-node lists from face topology.
///
/// Like rebuildCellConnectivity but also stores cellFaceNodes per cell,
/// needed by node ordering functions and writers.
std::vector<CellInfo> rebuildCellInfo(
    const std::vector<label>& faceOwner,
    const std::vector<label>& faceNeighbour,
    const std::vector<std::vector<localIdx>>& faceNodes,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nFaces
);

/// Order tet nodes: returns [base0, base1, base2, apex] (0-based).
std::vector<localIdx> orderTetNodes(const CellInfo& cell);

/// Order hex nodes: returns [bottom0..3, top0..3] (0-based).
std::vector<localIdx> orderHexNodes(const CellInfo& cell);

/// Order pyramid nodes: returns [base0..3, apex] (0-based).
std::vector<localIdx> orderPyramidNodes(const CellInfo& cell);

/// Order wedge/prism nodes: returns [bottom0..2, top0..2] (0-based).
std::vector<localIdx> orderWedgeNodes(const CellInfo& cell);

/// Extract cell connectivity from a VTK unstructured grid.
CellConnectivity extractCellConnectivity(vtkUnstructuredGrid* grid);


} // namespace NeoN::io
