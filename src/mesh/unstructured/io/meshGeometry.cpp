// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"

#include <cmath>
#include <vector>


namespace NeoN::io
{

namespace
{

Vec3 triangleArea(const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
    Vec3 e1 = p1 - p0;
    Vec3 e2 = p2 - p0;
    return Vec3 {
        0.5 * (e1[1] * e2[2] - e1[2] * e2[1]),
        0.5 * (e1[2] * e2[0] - e1[0] * e2[2]),
        0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
    };
}

scalar tetVolume(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3)
{
    Vec3 a = p1 - p0;
    Vec3 b = p2 - p0;
    Vec3 c = p3 - p0;
    return std::abs(
               a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
               + a[2] * (b[0] * c[1] - b[1] * c[0])
           )
         / 6.0;
}

} // anonymous namespace


MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells)
{
    localIdx nFaces = static_cast<localIdx>(topo.faceOwner.size());

    MeshGeometry geom;
    geom.faceCentres.resize(static_cast<std::size_t>(nFaces));
    geom.faceAreas.resize(static_cast<std::size_t>(nFaces));
    geom.magFaceAreas.resize(static_cast<std::size_t>(nFaces));

    // Compute face centres and face area vectors
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto fi = static_cast<std::size_t>(f);
        const auto& fn = topo.faceNodes[fi];
        localIdx nNodes = static_cast<localIdx>(fn.size());

        Vec3 centre {0.0, 0.0, 0.0};
        for (localIdx n = 0; n < nNodes; ++n)
        {
            centre = centre + points[static_cast<std::size_t>(fn[static_cast<std::size_t>(n)])];
        }
        centre = centre * (1.0 / static_cast<scalar>(nNodes));
        geom.faceCentres[fi] = centre;

        Vec3 area {0.0, 0.0, 0.0};
        for (localIdx n = 0; n < nNodes; ++n)
        {
            localIdx next = (n + 1) % nNodes;
            const Vec3& on = points[static_cast<std::size_t>(fn[static_cast<std::size_t>(n)])];
            const Vec3& pnext =
                points[static_cast<std::size_t>(fn[static_cast<std::size_t>(next)])];
            area = area + triangleArea(centre, on, pnext);
        }
        geom.faceAreas[fi] = area;
        geom.magFaceAreas[fi] = mag(area);
    }

    // Cell centres: geometric average of face centres touching the cell
    geom.cellVolumes.resize(static_cast<std::size_t>(nCells), 0.0);
    geom.cellCentres.resize(static_cast<std::size_t>(nCells), Vec3 {0.0, 0.0, 0.0});

    std::vector<int> facesPerCell(static_cast<std::size_t>(nCells), 0);
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto fi = static_cast<std::size_t>(f);
        auto ownIdx = static_cast<std::size_t>(topo.faceOwner[fi]);
        geom.cellCentres[ownIdx] = geom.cellCentres[ownIdx] + geom.faceCentres[fi];
        facesPerCell[ownIdx]++;

        if (f < topo.nInternalFaces)
        {
            auto neiIdx = static_cast<std::size_t>(topo.faceNeighbour[fi]);
            geom.cellCentres[neiIdx] = geom.cellCentres[neiIdx] + geom.faceCentres[fi];
            facesPerCell[neiIdx]++;
        }
    }
    for (localIdx c = 0; c < nCells; ++c)
    {
        auto ci = static_cast<std::size_t>(c);
        if (facesPerCell[ci] > 0)
        {
            geom.cellCentres[ci] =
                geom.cellCentres[ci] * (1.0 / static_cast<scalar>(facesPerCell[ci]));
        }
    }

    // Cell volumes via tetrahedral decomposition
    for (localIdx f = 0; f < nFaces; ++f)
    {
        auto fi = static_cast<std::size_t>(f);
        const auto& fn = topo.faceNodes[fi];
        localIdx nNodes = static_cast<localIdx>(fn.size());
        const Vec3& fc = geom.faceCentres[fi];

        auto ownerIdx = static_cast<std::size_t>(topo.faceOwner[fi]);
        const Vec3& cc = geom.cellCentres[ownerIdx];

        for (localIdx n = 0; n < nNodes; ++n)
        {
            localIdx next = (n + 1) % nNodes;
            const Vec3& on = points[static_cast<std::size_t>(fn[static_cast<std::size_t>(n)])];
            const Vec3& pnext =
                points[static_cast<std::size_t>(fn[static_cast<std::size_t>(next)])];
            geom.cellVolumes[ownerIdx] += tetVolume(cc, fc, on, pnext);
        }

        if (f < topo.nInternalFaces)
        {
            auto neiIdx = static_cast<std::size_t>(topo.faceNeighbour[fi]);
            const Vec3& ccNei = geom.cellCentres[neiIdx];
            for (localIdx n = 0; n < nNodes; ++n)
            {
                localIdx next = (n + 1) % nNodes;
                const Vec3& on = points[static_cast<std::size_t>(fn[static_cast<std::size_t>(n)])];
                const Vec3& pnext =
                    points[static_cast<std::size_t>(fn[static_cast<std::size_t>(next)])];
                geom.cellVolumes[neiIdx] += tetVolume(ccNei, fc, on, pnext);
            }
        }
    }

    return geom;
}


} // namespace NeoN::io
