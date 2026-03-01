// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"

#include <memory>
#include <vector>

namespace NeoN
{

UnstructuredMesh createUniform3DGrid(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar Lx, scalar Ly, scalar Lz
)
{
    const scalar dx = Lx / static_cast<scalar>(nx);
    const scalar dy = Ly / static_cast<scalar>(ny);
    const scalar dz = Lz / static_cast<scalar>(nz);

    const localIdx nPts = (nx + 1) * (ny + 1) * (nz + 1);
    const localIdx nCells = nx * ny * nz;

    // Face counts (all faces are quads)
    const localIdx nXInternal = (nx - 1) * ny * nz;
    const localIdx nYInternal = nx * (ny - 1) * nz;
    const localIdx nZInternal = nx * ny * (nz - 1);
    const localIdx nInternal = nXInternal + nYInternal + nZInternal;
    const localIdx nBndLeft = ny * nz;
    const localIdx nBndRight = ny * nz;
    const localIdx nBndBottom = nx * nz;
    const localIdx nBndTop = nx * nz;
    const localIdx nBndFront = nx * ny;
    const localIdx nBndBack = nx * ny;
    const localIdx nBoundary = nBndLeft + nBndRight + nBndBottom + nBndTop + nBndFront + nBndBack;
    const localIdx nFaces = nInternal + nBoundary;

    // Helper: cell index from grid coords
    auto cellIdx = [&](localIdx i, localIdx j, localIdx k) -> localIdx
    { return k * nx * ny + j * nx + i; };

    // Helper: point index from grid coords
    auto ptIdx = [&](localIdx i, localIdx j, localIdx k) -> localIdx
    { return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i; };

    // --- Points ---
    std::vector<Vec3> pts(static_cast<size_t>(nPts));
    for (localIdx k = 0; k <= nz; ++k)
        for (localIdx j = 0; j <= ny; ++j)
            for (localIdx i = 0; i <= nx; ++i)
                pts[static_cast<size_t>(ptIdx(i, j, k))] = {
                    static_cast<scalar>(i) * dx,
                    static_cast<scalar>(j) * dy,
                    static_cast<scalar>(k) * dz
                };
    vectorVector points(exec, pts);

    // --- Cell volumes and centres ---
    const scalar cellVol = dx * dy * dz;
    std::vector<scalar> vols(static_cast<size_t>(nCells), cellVol);
    scalarVector cellVolumes(exec, vols);

    std::vector<Vec3> centres(static_cast<size_t>(nCells));
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
            for (localIdx i = 0; i < nx; ++i)
                centres[static_cast<size_t>(cellIdx(i, j, k))] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
    vectorVector cellCentres(exec, centres);

    // --- Faces ---
    // Layout: [x-internal | y-internal | z-internal | left | right | bottom | top | front | back]
    std::vector<Vec3> fAreas(static_cast<size_t>(nFaces));
    std::vector<Vec3> fCentres(static_cast<size_t>(nFaces));
    std::vector<scalar> fMag(static_cast<size_t>(nFaces));
    std::vector<label> fOwner(static_cast<size_t>(nFaces));
    std::vector<label> fNeighbour(static_cast<size_t>(nInternal));

    localIdx faceId = 0;

    // X-normal internal faces, area = dy * dz
    const scalar xArea = dy * dz;
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
            for (localIdx i = 0; i < nx - 1; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {xArea, 0.0, 0.0};
                fCentres[fi] = {
                    static_cast<scalar>(i + 1) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
                fMag[fi] = xArea;
                fOwner[fi] = static_cast<label>(cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(cellIdx(i + 1, j, k));
                ++faceId;
            }

    // Y-normal internal faces, area = dx * dz
    const scalar yArea = dx * dz;
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny - 1; ++j)
            for (localIdx i = 0; i < nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, yArea, 0.0};
                fCentres[fi] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    static_cast<scalar>(j + 1) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
                fMag[fi] = yArea;
                fOwner[fi] = static_cast<label>(cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(cellIdx(i, j + 1, k));
                ++faceId;
            }

    // Z-normal internal faces, area = dx * dy
    const scalar zArea = dx * dy;
    for (localIdx k = 0; k < nz - 1; ++k)
        for (localIdx j = 0; j < ny; ++j)
            for (localIdx i = 0; i < nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, 0.0, zArea};
                fCentres[fi] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    static_cast<scalar>(k + 1) * dz
                };
                fMag[fi] = zArea;
                fOwner[fi] = static_cast<label>(cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(cellIdx(i, j, k + 1));
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

    // Left boundary (x=0)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
        {
            addBndFace(
                bndId,
                cellIdx(0, j, k),
                {-xArea, 0.0, 0.0},
                {0.0, (static_cast<scalar>(j) + 0.5) * dy, (static_cast<scalar>(k) + 0.5) * dz}
            );
            ++bndId;
            ++faceId;
        }

    // Right boundary (x=Lx)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
        {
            addBndFace(
                bndId,
                cellIdx(nx - 1, j, k),
                {xArea, 0.0, 0.0},
                {Lx, (static_cast<scalar>(j) + 0.5) * dy, (static_cast<scalar>(k) + 0.5) * dz}
            );
            ++bndId;
            ++faceId;
        }

    // Bottom boundary (y=0)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, 0, k),
                {0.0, -yArea, 0.0},
                {(static_cast<scalar>(i) + 0.5) * dx, 0.0, (static_cast<scalar>(k) + 0.5) * dz}
            );
            ++bndId;
            ++faceId;
        }

    // Top boundary (y=Ly)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, ny - 1, k),
                {0.0, yArea, 0.0},
                {(static_cast<scalar>(i) + 0.5) * dx, Ly, (static_cast<scalar>(k) + 0.5) * dz}
            );
            ++bndId;
            ++faceId;
        }

    // Front boundary (z=0)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, j, 0),
                {0.0, 0.0, -zArea},
                {(static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, 0.0}
            );
            ++bndId;
            ++faceId;
        }

    // Back boundary (z=Lz)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            addBndFace(
                bndId,
                cellIdx(i, j, nz - 1),
                {0.0, 0.0, zArea},
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

    // X-normal internal faces (at x=(i+1)*dx)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
            for (localIdx i = 0; i < nx - 1; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    ptIdx(i + 1, j, k),
                    ptIdx(i + 1, j + 1, k),
                    ptIdx(i + 1, j + 1, k + 1),
                    ptIdx(i + 1, j, k + 1)
                };
                ++fnId;
            }

    // Y-normal internal faces (at y=(j+1)*dy)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny - 1; ++j)
            for (localIdx i = 0; i < nx; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    ptIdx(i, j + 1, k),
                    ptIdx(i + 1, j + 1, k),
                    ptIdx(i + 1, j + 1, k + 1),
                    ptIdx(i, j + 1, k + 1)
                };
                ++fnId;
            }

    // Z-normal internal faces (at z=(k+1)*dz)
    for (localIdx k = 0; k < nz - 1; ++k)
        for (localIdx j = 0; j < ny; ++j)
            for (localIdx i = 0; i < nx; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    ptIdx(i, j, k + 1),
                    ptIdx(i + 1, j, k + 1),
                    ptIdx(i + 1, j + 1, k + 1),
                    ptIdx(i, j + 1, k + 1)
                };
                ++fnId;
            }

    // Left boundary (x=0)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(0, j, k), ptIdx(0, j + 1, k), ptIdx(0, j + 1, k + 1), ptIdx(0, j, k + 1)
            };
            ++fnId;
        }

    // Right boundary (x=Lx)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx j = 0; j < ny; ++j)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(nx, j, k), ptIdx(nx, j + 1, k), ptIdx(nx, j + 1, k + 1), ptIdx(nx, j, k + 1)
            };
            ++fnId;
        }

    // Bottom boundary (y=0)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, 0, k), ptIdx(i + 1, 0, k), ptIdx(i + 1, 0, k + 1), ptIdx(i, 0, k + 1)
            };
            ++fnId;
        }

    // Top boundary (y=Ly)
    for (localIdx k = 0; k < nz; ++k)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, ny, k), ptIdx(i + 1, ny, k), ptIdx(i + 1, ny, k + 1), ptIdx(i, ny, k + 1)
            };
            ++fnId;
        }

    // Front boundary (z=0)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, j, 0), ptIdx(i + 1, j, 0), ptIdx(i + 1, j + 1, 0), ptIdx(i, j + 1, 0)
            };
            ++fnId;
        }

    // Back boundary (z=Lz)
    for (localIdx j = 0; j < ny; ++j)
        for (localIdx i = 0; i < nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                ptIdx(i, j, nz), ptIdx(i + 1, j, nz), ptIdx(i + 1, j + 1, nz), ptIdx(i, j + 1, nz)
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
    mesh.stencilDB().insert(std::string("io::faceNodes"), faceNodesPtr);

    auto patchNames = std::make_shared<std::vector<std::string>>(
        std::vector<std::string> {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    );
    mesh.stencilDB().insert(std::string("io::patchNames"), patchNames);

    return mesh;
}

} // namespace NeoN
