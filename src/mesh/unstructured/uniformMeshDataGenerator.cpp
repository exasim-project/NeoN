// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/uniformMeshDataGenerator.hpp"

#include <memory>

namespace NeoN::detail
{

std::vector<Vec3> generatePoints(const MeshParams& p)
{
    const scalar dx = p.dx();
    const scalar dy = p.dy();
    const scalar dz = p.dz();
    const localIdx nPts = (p.nx + 1) * (p.ny + 1) * (p.nz + 1);
    std::vector<Vec3> pts(static_cast<size_t>(nPts));
    for (localIdx k = 0; k <= p.nz; ++k)
        for (localIdx j = 0; j <= p.ny; ++j)
            for (localIdx i = 0; i <= p.nx; ++i)
                pts[static_cast<size_t>(p.ptIdx(i, j, k))] = {
                    static_cast<scalar>(i) * dx,
                    static_cast<scalar>(j) * dy,
                    static_cast<scalar>(k) * dz
                };
    return pts;
}

CellData generateCellData(const MeshParams& p)
{
    const scalar dx = p.dx();
    const scalar dy = p.dy();
    const scalar dz = p.dz();
    const localIdx nCells = p.nx * p.ny * p.nz;
    const scalar cellVol = dx * dy * dz;
    std::vector<scalar> vols(static_cast<size_t>(nCells), cellVol);
    std::vector<Vec3> centers(static_cast<size_t>(nCells));
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
                centers[static_cast<size_t>(p.cellIdx(i, j, k))] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
    return {std::move(vols), std::move(centers)};
}

FaceData
generateInternalFaces(const MeshParams& p, const localIdx nInternalFaces, const localIdx nFaces)
{
    std::vector<Vec3> fAreas(static_cast<size_t>(nFaces));
    std::vector<Vec3> fCenters(static_cast<size_t>(nFaces));
    std::vector<scalar> fMag(static_cast<size_t>(nFaces));
    std::vector<label> fOwner(static_cast<size_t>(nFaces));
    std::vector<label> fNeighbour(static_cast<size_t>(nInternalFaces));

    localIdx faceId = 0;

    // Internal faces are ordered with x-normal faces first, then y-normal faces, then z-normal
    // faces. X-normal internal faces, area = dy * dz
    const scalar dx = p.dx();
    const scalar dy = p.dy();
    const scalar dz = p.dz();
    const scalar xArea = dy * dz;
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx - 1; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {xArea, 0.0, 0.0};
                fCenters[fi] = {
                    static_cast<scalar>(i + 1) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
                fMag[fi] = xArea;
                fOwner[fi] = p.cellIdx(i, j, k);
                fNeighbour[fi] = p.cellIdx(i + 1, j, k);
                ++faceId;
            }

    // Y-normal internal faces, area = dx * dz
    const scalar yArea = dx * dz;
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny - 1; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, yArea, 0.0};
                fCenters[fi] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    static_cast<scalar>(j + 1) * dy,
                    (static_cast<scalar>(k) + 0.5) * dz
                };
                fMag[fi] = yArea;
                fOwner[fi] = p.cellIdx(i, j, k);
                fNeighbour[fi] = p.cellIdx(i, j + 1, k);
                ++faceId;
            }

    // Z-normal internal faces, area = dx * dy
    const scalar zArea = dx * dy;
    for (localIdx k = 0; k < p.nz - 1; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
            for (localIdx i = 0; i < p.nx; ++i)
            {
                auto fi = static_cast<size_t>(faceId);
                fAreas[fi] = {0.0, 0.0, zArea};
                fCenters[fi] = {
                    (static_cast<scalar>(i) + 0.5) * dx,
                    (static_cast<scalar>(j) + 0.5) * dy,
                    static_cast<scalar>(k + 1) * dz
                };
                fMag[fi] = zArea;
                fOwner[fi] = p.cellIdx(i, j, k);
                fNeighbour[fi] = p.cellIdx(i, j, k + 1);
                ++faceId;
            }

    return {
        std::move(fAreas),
        std::move(fCenters),
        std::move(fMag),
        std::move(fOwner),
        std::move(fNeighbour)
        // nInternalFaces
    };
}

BoundaryMesh generateBoundaryData(
    const Executor exec,
    const int dim,
    const MeshParams& p,
    const std::vector<Vec3>& centers,
    const localIdx nInternalFaces,
    const localIdx nBoundaryFaces,
    const std::vector<localIdx> offset,
    FaceData& faces
)
{
    std::vector<label> bndFaceOwners(static_cast<size_t>(nBoundaryFaces));
    std::vector<Vec3> bndFaceCenters(static_cast<size_t>(nBoundaryFaces));
    std::vector<Vec3> bndCellCenters(static_cast<size_t>(nBoundaryFaces));
    std::vector<Vec3> bndFaceNormals(static_cast<size_t>(nBoundaryFaces));
    std::vector<scalar> bndFaceAreas(static_cast<size_t>(nBoundaryFaces));
    std::vector<Vec3> bndFaceUnitNormals(static_cast<size_t>(nBoundaryFaces));
    std::vector<Vec3> bndDelta(static_cast<size_t>(nBoundaryFaces));
    std::vector<scalar> bndWeights(static_cast<size_t>(nBoundaryFaces), 1.0);
    std::vector<scalar> bndDeltaCoeffs(static_cast<size_t>(nBoundaryFaces));

    localIdx faceId = nInternalFaces;

    auto addBndFace = [&](localIdx bndId, localIdx ci, Vec3 area, Vec3 faceCentre)
    {
        auto sz = static_cast<size_t>(bndId);
        auto fi = static_cast<size_t>(faceId);
        auto ciSizeT = static_cast<size_t>(ci);
        auto ciLabel = static_cast<label>(ci);
        scalar magA = mag(area);
        Vec3 normal = area * (1.0 / magA);
        Vec3 delta = faceCentre - centers[ciSizeT];

        faces.areas[fi] = area;
        faces.centers[fi] = faceCentre;
        faces.magnitudes[fi] = magA;
        faces.owner[fi] = ciLabel;

        bndFaceOwners[sz] = ciLabel;
        bndFaceCenters[sz] = faceCentre;
        bndCellCenters[sz] = centers[ciSizeT];
        bndFaceNormals[sz] = area;
        bndFaceAreas[sz] = magA;
        bndFaceUnitNormals[sz] = normal;
        bndDelta[sz] = delta;
        bndDeltaCoeffs[sz] = 1.0 / mag(delta);
    };

    localIdx bndId = 0;

    const scalar dx = p.dx();
    const scalar dy = p.dy();
    const scalar dz = p.dz();

    const scalar xArea = dy * dz;
    const scalar yArea = dx * dz;
    const scalar zArea = dx * dy;

    // Left boundary (x=0)
    for (localIdx k = 0; k < p.nz; ++k)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            addBndFace(
                bndId,
                p.cellIdx(0, j, k),
                {-xArea, 0.0, 0.0},
                {0.0, (static_cast<scalar>(j) + 0.5) * dy, (static_cast<scalar>(k) + 0.5) * dz}
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
                {p.Lx, (static_cast<scalar>(j) + 0.5) * dy, (static_cast<scalar>(k) + 0.5) * dz}
            );
            ++bndId;
            ++faceId;
        }

    if (dim > 1)
    {
        // Bottom boundary (y=0)
        for (localIdx k = 0; k < p.nz; ++k)
        {
            for (localIdx i = 0; i < p.nx; ++i)
            {
                addBndFace(
                    bndId,
                    p.cellIdx(i, 0, k),
                    {0.0, -yArea, 0.0},
                    {(static_cast<scalar>(i) + 0.5) * dx, 0.0, (static_cast<scalar>(k) + 0.5) * dz}
                );
                ++bndId;
                ++faceId;
            }
        }

        // Top boundary (y=Ly)
        for (localIdx k = 0; k < p.nz; ++k)
        {
            for (localIdx i = 0; i < p.nx; ++i)
            {
                addBndFace(
                    bndId,
                    p.cellIdx(i, p.ny - 1, k),
                    {0.0, yArea, 0.0},
                    {(static_cast<scalar>(i) + 0.5) * dx, p.Ly, (static_cast<scalar>(k) + 0.5) * dz}
                );
                ++bndId;
                ++faceId;
            }
        }
    }

    if (dim > 2)
    {
        // Front boundary (z=0)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            for (localIdx i = 0; i < p.nx; ++i)
            {
                addBndFace(
                    bndId,
                    p.cellIdx(i, j, 0),
                    {0.0, 0.0, -zArea},
                    {(static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, 0.0}
                );
                ++bndId;
                ++faceId;
            }
        }

        // Back boundary (z=Lz)
        for (localIdx j = 0; j < p.ny; ++j)
        {
            for (localIdx i = 0; i < p.nx; ++i)
            {
                addBndFace(
                    bndId,
                    p.cellIdx(i, j, p.nz - 1),
                    {0.0, 0.0, zArea},
                    {(static_cast<scalar>(i) + 0.5) * dx, (static_cast<scalar>(j) + 0.5) * dy, p.Lz}
                );
                ++bndId;
                ++faceId;
            }
        }
    }

    BoundaryMesh boundaryMesh(
        exec,
        {exec, bndFaceOwners},
        {exec, bndFaceCenters},
        {exec, bndCellCenters},
        {exec, bndFaceNormals},
        {exec, bndFaceAreas},
        {exec, bndFaceUnitNormals},
        {exec, bndDelta},
        {exec, bndWeights},
        {exec, bndDeltaCoeffs},
        offset,
        0,
        {}
    );

    return boundaryMesh;
}
} // namespace NeoN::detail
