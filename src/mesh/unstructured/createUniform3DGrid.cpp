// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "uniformGridHelpers.hpp"

namespace NeoN
{

UnstructuredMesh createUniform3DGrid(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar Lx, scalar Ly, scalar Lz
)
{
    detail::GridParams p {
        nx,
        ny,
        nz,
        Lx / static_cast<scalar>(nx),
        Ly / static_cast<scalar>(ny),
        Lz / static_cast<scalar>(nz),
        Lx,
        Ly,
        Lz
    };

    auto pts = detail::generatePoints(p);
    auto [vols, centres] = detail::generateCellData(p);
    auto faces = detail::generateInternalFaces(p);
    auto [boundaryMesh, nBoundary] = detail::generateBoundaryData(exec, p, centres, faces);

    const localIdx nCells = nx * ny * nz;
    const localIdx nFaces = faces.nInternal + nBoundary;

    auto faceNodesVec = detail::buildFaceNodes(p, nFaces);

    UnstructuredMesh mesh(
        vectorVector(exec, pts),
        scalarVector(exec, vols),
        vectorVector(exec, centres),
        {exec, faces.areas},
        {exec, faces.centres},
        {exec, faces.magnitudes},
        {exec, faces.owner},
        labelVector(exec, faces.neighbour),
        nCells,
        faces.nInternal,
        nBoundary,
        6,
        nFaces,
        boundaryMesh
    );

    detail::storeFaceNodesInStencilDB(mesh, faceNodesVec);

    return mesh;
}

} // namespace NeoN
