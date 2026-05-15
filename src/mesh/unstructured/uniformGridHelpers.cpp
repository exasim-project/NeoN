// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "uniformGridHelpers.hpp"

#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <memory>

namespace NeoN::detail
{

std::vector<Vec3> generatePoints(const GridParams& p)
{
    const localIdx nPts = (p.nx + 1) * (p.ny + 1) * (p.nz + 1);
    std::vector<Vec3> pts(static_cast<size_t>(nPts));
    for (localIdx k = 0; k <= p.nz; ++k)
        for (localIdx j = 0; j <= p.ny; ++j)
            for (localIdx i = 0; i <= p.nx; ++i)
                pts[static_cast<size_t>(p.ptIdx(i, j, k))] = {
                    static_cast<scalar>(i) * p.dx,
                    static_cast<scalar>(j) * p.dy,
                    static_cast<scalar>(k) * p.dz
                };
    return pts;
}

CellData generateCellData(const GridParams& p)
{
    const localIdx nCells = p.nx * p.ny * p.nz;
    const scalar cellVol = p.dx * p.dy * p.dz;
    std::vector<scalar> vols(static_cast<size_t>(nCells), cellVol);
    std::vector<Vec3> centres(static_cast<size_t>(nCells));
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
                centres[static_cast<size_t>(p.cellIdx(i, j, k))] = {
                    (static_cast<scalar>(i) + 0.5) * p.dx,
                    (static_cast<scalar>(j) + 0.5) * p.dy,
                    (static_cast<scalar>(k) + 0.5) * p.dz
                };
    return {std::move(vols), std::move(centres)};
}

FaceData generateInternalFaces(const GridParams& p)
{
    const localIdx nXInternal = (p.nx - 1) * p.ny * p.nz;
    const localIdx nYInternal = p.nx * (p.ny - 1) * p.nz;
    const localIdx nZInternal = p.nx * p.ny * (p.nz - 1);
    const localIdx nInternal = nXInternal + nYInternal + nZInternal;

    const localIdx nBndLeft = p.ny * p.nz;
    const localIdx nBndRight = p.ny * p.nz;
    const localIdx nBndBottom = p.nx * p.nz;
    const localIdx nBndTop = p.nx * p.nz;
    const localIdx nBndFront = p.nx * p.ny;
    const localIdx nBndBack = p.nx * p.ny;
    const localIdx nBoundary = nBndLeft + nBndRight + nBndBottom + nBndTop + nBndFront + nBndBack;
    const localIdx nFaces = nInternal + nBoundary;

    std::vector<Vec3> fAreas(static_cast<size_t>(nFaces));
    std::vector<Vec3> fCentres(static_cast<size_t>(nFaces));
    std::vector<scalar> fMag(static_cast<size_t>(nFaces));
    std::vector<label> fOwner(static_cast<size_t>(nFaces));
    std::vector<label> fNeighbour(static_cast<size_t>(nInternal));

    localIdx faceId = 0;

    // X-normal internal faces, area = dy * dz
    const scalar xArea = p.dy * p.dz;
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx - 1; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {xArea, 0.0, 0.0};
                fCentres[fi] = {
                    static_cast<scalar>(i + 1) * p.dx,
                    (static_cast<scalar>(j) + 0.5) * p.dy,
                    (static_cast<scalar>(k) + 0.5) * p.dz
                };
                fMag[fi] = xArea;
                fOwner[fi] = static_cast<label>(p.cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(p.cellIdx(i + 1, j, k));
                ++faceId;
            }

    // Y-normal internal faces, area = dx * dz
    const scalar yArea = p.dx * p.dz;
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny - 1; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, yArea, 0.0};
                fCentres[fi] = {
                    (static_cast<scalar>(i) + 0.5) * p.dx,
                    static_cast<scalar>(j + 1) * p.dy,
                    (static_cast<scalar>(k) + 0.5) * p.dz
                };
                fMag[fi] = yArea;
                fOwner[fi] = static_cast<label>(p.cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(p.cellIdx(i, j + 1, k));
                ++faceId;
            }

    // Z-normal internal faces, area = dx * dy
    const scalar zArea = p.dx * p.dy;
    for (localIdx k = 0; k < p.nz - 1; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, 0.0, zArea};
                fCentres[fi] = {
                    (static_cast<scalar>(i) + 0.5) * p.dx,
                    (static_cast<scalar>(j) + 0.5) * p.dy,
                    static_cast<scalar>(k + 1) * p.dz
                };
                fMag[fi] = zArea;
                fOwner[fi] = static_cast<label>(p.cellIdx(i, j, k));
                fNeighbour[fi] = static_cast<label>(p.cellIdx(i, j, k + 1));
                ++faceId;
            }

    return {
        std::move(fAreas),
        std::move(fCentres),
        std::move(fMag),
        std::move(fOwner),
        std::move(fNeighbour),
        nInternal
    };
}

BoundaryData generateBoundaryData(
    const Executor exec, const GridParams& p, const std::vector<Vec3>& centres, FaceData& faces
)
{
    const localIdx nBndLeft = p.ny * p.nz;
    const localIdx nBndRight = p.ny * p.nz;
    const localIdx nBndBottom = p.nx * p.nz;
    const localIdx nBndTop = p.nx * p.nz;
    const localIdx nBndFront = p.nx * p.ny;
    const localIdx nBndBack = p.nx * p.ny;
    const localIdx nBoundary = nBndLeft + nBndRight + nBndBottom + nBndTop + nBndFront + nBndBack;

    std::vector<label> bndFaceCells(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndCf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndCn(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndSf(static_cast<size_t>(nBoundary));
    std::vector<scalar> bndMagSf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndNf(static_cast<size_t>(nBoundary));
    std::vector<Vec3> bndDelta(static_cast<size_t>(nBoundary));
    std::vector<scalar> bndWeights(static_cast<size_t>(nBoundary), 1.0);
    std::vector<scalar> bndDeltaCoeffs(static_cast<size_t>(nBoundary));

    localIdx faceId = faces.nInternal;

    auto addBndFace = [&](localIdx bndId, localIdx ci, Vec3 area, Vec3 faceCentre)
    {
        auto sz = static_cast<size_t>(bndId);
        auto fi = static_cast<size_t>(faceId);
        scalar magA = mag(area);
        Vec3 normal = area * (1.0 / magA);
        Vec3 delta = faceCentre - centres[static_cast<size_t>(ci)];

        faces.areas[fi] = area;
        faces.centres[fi] = faceCentre;
        faces.magnitudes[fi] = magA;
        faces.owner[fi] = static_cast<label>(ci);

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

    const scalar xArea = p.dy * p.dz;
    const scalar yArea = p.dx * p.dz;
    const scalar zArea = p.dx * p.dy;

    // Left boundary (x=0)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            addBndFace(
                bndId,
                p.cellIdx(0, j, k),
                {-xArea, 0.0, 0.0},
                {0.0, (static_cast<scalar>(j) + 0.5) * p.dy, (static_cast<scalar>(k) + 0.5) * p.dz}
            );
            ++bndId;
            ++faceId;
        }

    // Right boundary (x=Lx)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            addBndFace(
                bndId,
                p.cellIdx(p.nx - 1, j, k),
                {xArea, 0.0, 0.0},
                {p.Lx, (static_cast<scalar>(j) + 0.5) * p.dy, (static_cast<scalar>(k) + 0.5) * p.dz}
            );
            ++bndId;
            ++faceId;
        }

    // Bottom boundary (y=0)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            addBndFace(
                bndId,
                p.cellIdx(i, 0, k),
                {0.0, -yArea, 0.0},
                {(static_cast<scalar>(i) + 0.5) * p.dx, 0.0, (static_cast<scalar>(k) + 0.5) * p.dz}
            );
            ++bndId;
            ++faceId;
        }

    // Top boundary (y=Ly)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            addBndFace(
                bndId,
                p.cellIdx(i, p.ny - 1, k),
                {0.0, yArea, 0.0},
                {(static_cast<scalar>(i) + 0.5) * p.dx, p.Ly, (static_cast<scalar>(k) + 0.5) * p.dz}
            );
            ++bndId;
            ++faceId;
        }

    // Front boundary (z=0)
    for (localIdx j = 0; j < p.ny; ++j)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            addBndFace(
                bndId,
                p.cellIdx(i, j, 0),
                {0.0, 0.0, -zArea},
                {(static_cast<scalar>(i) + 0.5) * p.dx, (static_cast<scalar>(j) + 0.5) * p.dy, 0.0}
            );
            ++bndId;
            ++faceId;
        }

    // Back boundary (z=Lz)
    for (localIdx j = 0; j < p.ny; ++j)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            addBndFace(
                bndId,
                p.cellIdx(i, j, p.nz - 1),
                {0.0, 0.0, zArea},
                {(static_cast<scalar>(i) + 0.5) * p.dx, (static_cast<scalar>(j) + 0.5) * p.dy, p.Lz}
            );
            ++bndId;
            ++faceId;
        }

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

    return {std::move(boundaryMesh), nBoundary};
}

std::vector<std::vector<localIdx>> buildFaceNodes(const GridParams& p, localIdx nFaces)
{
    std::vector<std::vector<localIdx>> faceNodesVec(static_cast<size_t>(nFaces));
    localIdx fnId = 0;

    // X-normal internal faces
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx - 1; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    p.ptIdx(i + 1, j, k),
                    p.ptIdx(i + 1, j + 1, k),
                    p.ptIdx(i + 1, j + 1, k + 1),
                    p.ptIdx(i + 1, j, k + 1)
                };
                ++fnId;
            }

    // Y-normal internal faces
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny - 1; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    p.ptIdx(i, j + 1, k),
                    p.ptIdx(i + 1, j + 1, k),
                    p.ptIdx(i + 1, j + 1, k + 1),
                    p.ptIdx(i, j + 1, k + 1)
                };
                ++fnId;
            }

    // Z-normal internal faces
    for (localIdx k = 0; k < p.nz - 1; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                faceNodesVec[static_cast<size_t>(fnId)] = {
                    p.ptIdx(i, j, k + 1),
                    p.ptIdx(i + 1, j, k + 1),
                    p.ptIdx(i + 1, j + 1, k + 1),
                    p.ptIdx(i, j + 1, k + 1)
                };
                ++fnId;
            }

    // Left boundary (x=0)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(0, j, k),
                p.ptIdx(0, j + 1, k),
                p.ptIdx(0, j + 1, k + 1),
                p.ptIdx(0, j, k + 1)
            };
            ++fnId;
        }

    // Right boundary (x=Lx)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(p.nx, j, k),
                p.ptIdx(p.nx, j + 1, k),
                p.ptIdx(p.nx, j + 1, k + 1),
                p.ptIdx(p.nx, j, k + 1)
            };
            ++fnId;
        }

    // Bottom boundary (y=0)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(i, 0, k),
                p.ptIdx(i + 1, 0, k),
                p.ptIdx(i + 1, 0, k + 1),
                p.ptIdx(i, 0, k + 1)
            };
            ++fnId;
        }

    // Top boundary (y=Ly)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(i, p.ny, k),
                p.ptIdx(i + 1, p.ny, k),
                p.ptIdx(i + 1, p.ny, k + 1),
                p.ptIdx(i, p.ny, k + 1)
            };
            ++fnId;
        }

    // Front boundary (z=0)
    for (localIdx j = 0; j < p.ny; ++j)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(i, j, 0),
                p.ptIdx(i + 1, j, 0),
                p.ptIdx(i + 1, j + 1, 0),
                p.ptIdx(i, j + 1, 0)
            };
            ++fnId;
        }

    // Back boundary (z=Lz)
    for (localIdx j = 0; j < p.ny; ++j)
        for (localIdx i = 0; i < p.nx; ++i)
        {
            faceNodesVec[static_cast<size_t>(fnId)] = {
                p.ptIdx(i, j, p.nz),
                p.ptIdx(i + 1, j, p.nz),
                p.ptIdx(i + 1, j + 1, p.nz),
                p.ptIdx(i, j + 1, p.nz)
            };
            ++fnId;
        }

    return faceNodesVec;
}

void storeFaceNodesInStencilDB(
    UnstructuredMesh& mesh, const std::vector<std::vector<localIdx>>& faceNodesVec
)
{
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
}

} // namespace NeoN::detail
