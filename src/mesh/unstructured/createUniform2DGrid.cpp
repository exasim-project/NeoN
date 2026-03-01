// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"

#include <memory>
#include <vector>

namespace NeoN
{

UnstructuredMesh
createUniform2DGrid(const Executor exec, localIdx nx, localIdx ny, scalar Lx, scalar Ly)
{
    // OpenFOAM-style 2D mesh: hex cells one cell thick in z.
    // Domain: [0,Lx] x [0,Ly] x [0,Lz], Lz = 1.0
    const scalar dx = Lx / static_cast<scalar>(nx);
    const scalar dy = Ly / static_cast<scalar>(ny);
    const scalar Lz = 1.0;
    const localIdx nPtsPerPlane = (nx + 1) * (ny + 1);
    const localIdx nPts = nPtsPerPlane * 2; // two z-planes
    const localIdx nCells = nx * ny;

    // Face counts (all faces are quads)
    const localIdx nVerticalInternal = (nx - 1) * ny;   // y-z plane faces between columns
    const localIdx nHorizontalInternal = nx * (ny - 1); // x-z plane faces between rows
    const localIdx nInternal = nVerticalInternal + nHorizontalInternal;
    const localIdx nBndLeft = ny;
    const localIdx nBndRight = ny;
    const localIdx nBndBottom = nx;
    const localIdx nBndTop = nx;
    const localIdx nBndFront = nCells; // z=0 plane
    const localIdx nBndBack = nCells;  // z=Lz plane
    const localIdx nBoundary = nBndLeft + nBndRight + nBndBottom + nBndTop + nBndFront + nBndBack;
    const localIdx nFaces = nInternal + nBoundary;

    // Helper: cell index from grid coords
    auto cellIdx = [&](localIdx i, localIdx j) -> localIdx { return j * nx + i; };

    // Helper: point index from grid coords and z-plane
    // k=0 bottom (z=0), k=1 top (z=Lz)
    auto ptIdx = [&](localIdx i, localIdx j, localIdx k) -> localIdx
    { return k * nPtsPerPlane + j * (nx + 1) + i; };

    // --- Points (two z-planes) ---
    std::vector<Vec3> pts(static_cast<size_t>(nPts));
    for (localIdx k = 0; k <= 1; ++k)
        for (localIdx j = 0; j <= ny; ++j)
            for (localIdx i = 0; i <= nx; ++i)
                pts[static_cast<size_t>(ptIdx(i, j, k))] = {
                    static_cast<scalar>(i) * dx,
                    static_cast<scalar>(j) * dy,
                    static_cast<scalar>(k) * Lz
                };
    vectorVector points(exec, pts);

    // --- Cell volumes and centres ---
    const scalar cellVol = dx * dy * Lz;
    std::vector<scalar> vols(static_cast<size_t>(nCells), cellVol);
    scalarVector cellVolumes(exec, vols);

    std::vector<Vec3> centres(static_cast<size_t>(nCells));
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
            centres[static_cast<size_t>(cellIdx(i, j))] = {
                (static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, 0.5 * Lz
            };
    vectorVector cellCentres(exec, centres);

    // --- Faces ---
    // Layout: [vert-internal | horiz-internal | left | right | bottom | top | front | back]
    // All faces are quads (4 nodes). Face areas are actual areas (dy*Lz, dx*Lz, dx*dy).
    std::vector<Vec3> fAreas(static_cast<size_t>(nFaces));
    std::vector<Vec3> fCentres(static_cast<size_t>(nFaces));
    std::vector<scalar> fMag(static_cast<size_t>(nFaces));
    std::vector<label> fOwner(static_cast<size_t>(nFaces));
    std::vector<label> fNeighbour(static_cast<size_t>(nInternal));

    localIdx faceId = 0;

    // Vertical internal faces (y-z plane, normal in +x), area = dy * Lz
    const scalar vertArea = dy * Lz;
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx - 1; ++i)
        {
            auto fi = static_cast<size_t>(faceId);
            fAreas[fi] = {vertArea, 0.0, 0.0};
            fCentres[fi] = {
                static_cast<scalar>(i + 1) * dx, (static_cast<scalar>(j) + 0.5) * dy, 0.5 * Lz
            };
            fMag[fi] = vertArea;
            fOwner[fi] = static_cast<label>(cellIdx(i, j));
            fNeighbour[fi] = static_cast<label>(cellIdx(i + 1, j));
            ++faceId;
        }

    // Horizontal internal faces (x-z plane, normal in +y), area = dx * Lz
    const scalar horizArea = dx * Lz;
    for (localIdx j = 0; j < ny - 1; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            auto fi = static_cast<size_t>(faceId);
            fAreas[fi] = {0.0, horizArea, 0.0};
            fCentres[fi] = {
                (static_cast<scalar>(i) + 0.5) * dx, static_cast<scalar>(j + 1) * dy, 0.5 * Lz
            };
            fMag[fi] = horizArea;
            fOwner[fi] = static_cast<label>(cellIdx(i, j));
            fNeighbour[fi] = static_cast<label>(cellIdx(i, j + 1));
            ++faceId;
        }

    // --- Boundary faces ---
    std::vector<label> bndFaceCells(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndCf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndCn(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndSf(static_cast<size_t>(nBoundary));
    std::vector<scalar> bndMagSf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndNf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndDelta(static_cast<size_t>(nBoundary));
    std::vector<scalar> bndWeights(static_cast<size_t>(nBoundary), 1.0);
    std::vector<scalar> bndDeltaCoeffs(static_cast<size_t>(nBoundary));

    // Helper to fill one boundary face
    auto addBndFace = [&](localIdx bndId, localIdx ci, Vec3 area, Vec3 faceCentre)
    {
        auto sz = static_cast<size_t>(bndId);
        auto fi = static_cast<size_t>(faceId);
        scalar magA = mag(area);
        Vec3 normal = area * (1.0 / magA);
        Vec3 delta = faceCentre - centres[static_cast<size_t>(ci)];

        fAreas[fi] = area;
        fCentres[fi] = faceCentre;
        fMag[fi] = magA;
        fOwner[fi] = static_cast<label>(ci);

        bndFaceCells[sz] = static_cast<label>(ci);
        bndCf[sz] = faceCentre;
        bndCn[sz] = centres[static_cast<size_t>(ci)];
        bndSf[sz] = area;
        bndMagSf[sz] = magA;
        bndNf[sz] = normal;
        bndDelta[sz] = delta;
        bndDeltaCoeffs[sz] = 1.0 / mag(delta);
    };

    localIdx bndId = 0;

    // Left boundary (x=0), normal = {-1,0,0}, area = dy*Lz
    for (localIdx j = 0; j < ny; ++j)
    {
        addBndFace(
            bndId,
            cellIdx(0, j),
            {-vertArea, 0.0, 0.0},
            {0.0, (static_cast<scalar>(j) + 0.5) * dy, 0.5 * Lz}
        );
        ++bndId;
        ++faceId;
    }

    // Right boundary (x=Lx), normal = {1,0,0}, area = dy*Lz
    for (localIdx j = 0; j < ny; ++j)
    {
        addBndFace(
            bndId,
            cellIdx(nx - 1, j),
            {vertArea, 0.0, 0.0},
            {Lx, (static_cast<scalar>(j) + 0.5) * dy, 0.5 * Lz}
        );
        ++bndId;
        ++faceId;
    }

    // Bottom boundary (y=0), normal = {0,-1,0}, area = dx*Lz
    for (localIdx i = 0; i < nx; ++i)
    {
        addBndFace(
            bndId,
            cellIdx(i, 0),
            {0.0, -horizArea, 0.0},
            {(static_cast<scalar>(i) + 0.5) * dx, 0.0, 0.5 * Lz}
        );
        ++bndId;
        ++faceId;
    }

    // Top boundary (y=Ly), normal = {0,1,0}, area = dx*Lz
    for (localIdx i = 0; i < nx; ++i)
    {
        addBndFace(
            bndId,
            cellIdx(i, ny - 1),
            {0.0, horizArea, 0.0},
            {(static_cast<scalar>(i) + 0.5) * dx, Ly, 0.5 * Lz}
        );
        ++bndId;
        ++faceId;
    }

    // Front boundary (z=0), normal = {0,0,-1}, area = dx*dy
    const scalar frontBackArea = dx * dy;
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, j),
                {0.0, 0.0, -frontBackArea},
                {(static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, 0.0}
            );
            ++bndId;
            ++faceId;
        }

    // Back boundary (z=Lz), normal = {0,0,1}, area = dx*dy
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, j),
                {0.0, 0.0, frontBackArea},
                {(static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, Lz}
            );
            ++bndId;
            ++faceId;
        }

    // Boundary patch offsets: left | right | bottom | top | front | back
    std::vector<localIdx> offset = {
        0,
        nBndLeft,
        nBndLeft + nBndRight,
        nBndLeft + nBndRight + nBndBottom,
        nBndLeft + nBndRight + nBndBottom + nBndTop,
        nBndLeft + nBndRight + nBndBottom + nBndTop + nBndFront,
        nBoundary
    };

    BoundaryMesh boundaryMesh(
        exec,
        {exec, bndFaceCells},
        {exec, bndCf},
        {exec, bndCn},
        {exec, bndSf},
        {exec, bndMagSf},
        {exec, bndNf},
        {exec, bndDelta},
        {exec, bndWeights},
        {exec, bndDeltaCoeffs},
        offset
    );

    // --- Face node connectivity for IO writers ---
    // Every face is a quad (4 nodes).
    std::vector<std::vector<localIdx>> faceNodesVec(static_cast<size_t>(nFaces));

    localIdx fnId = 0;

    // Vertical internal faces (y-z plane at x=(i+1)*dx)
    // Nodes: bottom-left, bottom-right (top z), top-right, top-left
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx - 1; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i + 1, j, 0),
                ptIdx(i + 1, j + 1, 0),
                ptIdx(i + 1, j + 1, 1),
                ptIdx(i + 1, j, 1)
            };
            ++fnId;
        }

    // Horizontal internal faces (x-z plane at y=(j+1)*dy)
    for (localIdx j = 0; j < ny - 1; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, j + 1, 0),
                ptIdx(i + 1, j + 1, 0),
                ptIdx(i + 1, j + 1, 1),
                ptIdx(i, j + 1, 1)
            };
            ++fnId;
        }

    // Left boundary (x=0, y-z plane)
    for (localIdx j = 0; j < ny; ++j)
    {
        faceNodesVec[static_cast<size_t>(fnId)] = {
            ptIdx(0, j, 0), ptIdx(0, j + 1, 0), ptIdx(0, j + 1, 1), ptIdx(0, j, 1)
        };
        ++fnId;
    }

    // Right boundary (x=Lx, y-z plane)
    for (localIdx j = 0; j < ny; ++j)
    {
        faceNodesVec[static_cast<size_t>(fnId)] = {
            ptIdx(nx, j, 0), ptIdx(nx, j + 1, 0), ptIdx(nx, j + 1, 1), ptIdx(nx, j, 1)
        };
        ++fnId;
    }

    // Bottom boundary (y=0, x-z plane)
    for (localIdx i = 0; i < nx; ++i)
    {
        faceNodesVec[static_cast<size_t>(fnId)] = {
            ptIdx(i, 0, 0), ptIdx(i + 1, 0, 0), ptIdx(i + 1, 0, 1), ptIdx(i, 0, 1)
        };
        ++fnId;
    }

    // Top boundary (y=Ly, x-z plane)
    for (localIdx i = 0; i < nx; ++i)
    {
        faceNodesVec[static_cast<size_t>(fnId)] = {
            ptIdx(i, ny, 0), ptIdx(i + 1, ny, 0), ptIdx(i + 1, ny, 1), ptIdx(i, ny, 1)
        };
        ++fnId;
    }

    // Front boundary (z=0, x-y plane)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, j, 0), ptIdx(i + 1, j, 0), ptIdx(i + 1, j + 1, 0), ptIdx(i, j + 1, 0)
            };
            ++fnId;
        }

    // Back boundary (z=Lz, x-y plane)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, j, 1), ptIdx(i + 1, j, 1), ptIdx(i + 1, j + 1, 1), ptIdx(i, j + 1, 1)
            };
            ++fnId;
        }

    UnstructuredMesh mesh(
        points,
        cellVolumes,
        cellCentres,
        {exec, fAreas},
        {exec, fCentres},
        {exec, fMag},
        {exec, fOwner},
        labelVector(exec, fNeighbour),
        nCells,
        nInternal,
        nBoundary,
        6,
        nFaces,
        boundaryMesh
    );

    // Convert to SegmentedVector for stencilDB (new canonical type for io::faceNodes)
    std::vector<localIdx> fnValues, fnOffsets;
    fnOffsets.push_back(0);
    for (const auto& face : faceNodesVec)
    {
        fnValues.insert(fnValues.end(), face.begin(), face.end());
        fnOffsets.push_back(
            static_cast<localIdx>(fnOffsets.back() + static_cast<localIdx>(face.size()))
        );
    }
    SerialExecutor serial;
    auto faceNodesPtr = std::make_shared<SegmentedVector<localIdx, localIdx>>(
        Vector<localIdx>(serial, fnValues), Vector<localIdx>(serial, fnOffsets)
    );
    mesh.stencilDB().insert(std::string(io::stencilFaceNodes), faceNodesPtr);

    auto patchNames = std::make_shared<std::vector<std::string>>(
        std::vector<std::string> {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    );
    mesh.stencilDB().insert(std::string(io::stencilPatchNames), patchNames);

    return mesh;
}

} // namespace NeoN
