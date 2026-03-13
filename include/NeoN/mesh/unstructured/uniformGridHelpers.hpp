// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"

#include <vector>

namespace NeoN::detail
{

struct GridParams
{
    localIdx nx, ny, nz;
    scalar dx, dy, dz;
    scalar Lx, Ly, Lz;

    localIdx cellIdx(localIdx i, localIdx j, localIdx k) const { return k * nx * ny + j * nx + i; }

    localIdx ptIdx(localIdx i, localIdx j, localIdx k) const
    {
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
    }
};

struct CellData
{
    std::vector<scalar> volumes;
    std::vector<Vec3> centres;
};

struct FaceData
{
    std::vector<Vec3> areas;
    std::vector<Vec3> centres;
    std::vector<scalar> magnitudes;
    std::vector<label> owner;
    std::vector<label> neighbour;
    localIdx nInternal;
};

struct BoundaryData
{
    BoundaryMesh mesh;
    localIdx nBoundary;
};

std::vector<Vec3> generatePoints(const GridParams& p);

CellData generateCellData(const GridParams& p);

FaceData generateInternalFaces(const GridParams& p);

BoundaryData generateBoundaryData(
    const Executor exec, const GridParams& p, const std::vector<Vec3>& centres, FaceData& faces
);

std::vector<std::vector<localIdx>> buildFaceNodes(const GridParams& p, localIdx nFaces);

void storeFaceNodesInStencilDB(
    UnstructuredMesh& mesh, const std::vector<std::vector<localIdx>>& faceNodesVec
);

} // namespace NeoN::detail
