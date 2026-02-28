// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"

#include <vector>

namespace NeoN::io
{

/// Geometric quantities derived from mesh topology and point coordinates.
struct MeshGeometry
{
    std::vector<scalar> cellVolumes;
    std::vector<Vec3> cellCentres;
    std::vector<Vec3> faceAreas;
    std::vector<Vec3> faceCentres;
    std::vector<scalar> magFaceAreas;
};


/// Compute all geometric quantities from points and face topology.
///
/// Face areas via triangulation from centroid, cell centres as average
/// of touching face centres, cell volumes via tetrahedral decomposition.
MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells);


} // namespace NeoN::io
