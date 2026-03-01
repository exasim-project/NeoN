// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include <cmath>
#include <vector>


namespace NeoN::io
{

namespace
{

NEON_INLINE_FUNCTION
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

NEON_INLINE_FUNCTION
scalar tetVolume(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3)
{
    Vec3 a = p1 - p0;
    Vec3 b = p2 - p0;
    Vec3 c = p3 - p0;
    scalar det = a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
               + a[2] * (b[0] * c[1] - b[1] * c[0]);
    return Kokkos::fabs(det) / 6.0;
}

} // anonymous namespace


Vector<Vec3> computeFaceCentres(
    const Executor& exec, const Vector<Vec3>& points, SegmentedVector<localIdx, localIdx>& faceNodes
)
{
    localIdx nFaces = faceNodes.numSegments();
    Vector<Vec3> faceCentres(exec, nFaces, Vec3 {0.0, 0.0, 0.0});

    auto fcView = faceCentres.view();
    auto ptsView = points.view();
    auto fnView = faceNodes.view();

    parallelFor(
        exec,
        {0, nFaces},
        NEON_LAMBDA(const localIdx f) {
            auto [start, end] = fnView.bounds(f);
            Vec3 centre {0.0, 0.0, 0.0};
            localIdx nNodes = end - start;
            for (localIdx n = start; n < end; ++n)
            {
                centre = centre + ptsView[fnView.values[n]];
            }
            fcView[f] = centre * (1.0 / static_cast<scalar>(nNodes));
        },
        "computeFaceCentres"
    );

    return faceCentres;
}


Vector<Vec3> computeFaceAreas(
    const Executor& exec,
    const Vector<Vec3>& points,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    const Vector<Vec3>& faceCentres
)
{
    localIdx nFaces = faceNodes.numSegments();
    Vector<Vec3> faceAreas(exec, nFaces, Vec3 {0.0, 0.0, 0.0});

    auto faView = faceAreas.view();
    auto ptsView = points.view();
    auto fnView = faceNodes.view();
    auto fcView = faceCentres.view();

    parallelFor(
        exec,
        {0, nFaces},
        NEON_LAMBDA(const localIdx f) {
            auto [start, end] = fnView.bounds(f);
            localIdx nNodes = end - start;
            Vec3 area {0.0, 0.0, 0.0};
            Vec3 centre = fcView[f];
            for (localIdx n = 0; n < nNodes; ++n)
            {
                localIdx curr = start + n;
                localIdx next = start + ((n + 1) % nNodes);
                area = area
                     + triangleArea(
                           centre, ptsView[fnView.values[curr]], ptsView[fnView.values[next]]
                     );
            }
            faView[f] = area;
        },
        "computeFaceAreas"
    );

    return faceAreas;
}


Vector<scalar> computeMagFaceAreas(const Executor& exec, const Vector<Vec3>& faceAreas)
{
    localIdx nFaces = faceAreas.size();
    Vector<scalar> magFaceAreas(exec, nFaces);

    auto magView = magFaceAreas.view();
    auto faView = faceAreas.view();

    parallelFor(
        exec,
        {0, nFaces},
        NEON_LAMBDA(const localIdx f) { magView[f] = mag(faView[f]); },
        "computeMagFaceAreas"
    );

    return magFaceAreas;
}


SegmentedVector<localIdx, localIdx> buildCellToFaceMapping(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    localIdx nInternalFaces,
    localIdx nCells
)
{
    localIdx nFaces = faceOwner.size();

    // Step A: Count faces per cell
    Vector<localIdx> facesPerCell(exec, nCells, localIdx(0));
    auto fpcView = facesPerCell.view();
    auto ownerView = faceOwner.view();
    auto neiView = faceNeighbour.view();

    // Internal faces contribute to both owner and neighbour
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx f) {
            Kokkos::atomic_increment(&fpcView[ownerView[f]]);
            Kokkos::atomic_increment(&fpcView[neiView[f]]);
        },
        "countFacesPerCell_internal"
    );

    // Boundary faces contribute to owner only
    parallelFor(
        exec,
        {nInternalFaces, nFaces},
        NEON_LAMBDA(const localIdx f) { Kokkos::atomic_increment(&fpcView[ownerView[f]]); },
        "countFacesPerCell_boundary"
    );

    SegmentedVector<localIdx, localIdx> cellFaces(facesPerCell);

    fill(facesPerCell, localIdx(0));
    auto [cfValues, cfSegments] = cellFaces.views();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx f) {
            auto own = ownerView[f];
            auto pos = Kokkos::atomic_fetch_add(&fpcView[own], localIdx(1));
            cfValues[cfSegments[own] + pos] = f;
            auto nei = neiView[f];
            pos = Kokkos::atomic_fetch_add(&fpcView[nei], localIdx(1));
            cfValues[cfSegments[nei] + pos] = f;
        },
        "fillCellFaces_internal"
    );

    parallelFor(
        exec,
        {nInternalFaces, nFaces},
        NEON_LAMBDA(const localIdx f) {
            auto own = ownerView[f];
            auto pos = Kokkos::atomic_fetch_add(&fpcView[own], localIdx(1));
            cfValues[cfSegments[own] + pos] = f;
        },
        "fillCellFaces_boundary"
    );

    return cellFaces;
}


Vector<Vec3> computeCellCentres(
    const Executor& exec,
    const Vector<Vec3>& faceCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
)
{
    Vector<Vec3> cellCentres(exec, nCells, Vec3 {0.0, 0.0, 0.0});

    auto ccView = cellCentres.view();
    auto fcView = faceCentres.view();
    auto cfView = cellFaces.view();

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) {
            auto [start, end] = cfView.bounds(c);
            Vec3 sum {0.0, 0.0, 0.0};
            for (localIdx i = start; i < end; ++i)
            {
                sum = sum + fcView[cfView.values[i]];
            }
            ccView[c] = sum * (1.0 / static_cast<scalar>(end - start));
        },
        "computeCellCentres"
    );

    return cellCentres;
}


Vector<scalar> computeCellVolumes(
    const Executor& exec,
    const Vector<Vec3>& points,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    const Vector<Vec3>& faceCentres,
    const Vector<Vec3>& cellCentres,
    SegmentedVector<localIdx, localIdx>& cellFaces,
    localIdx nCells
)
{
    Vector<scalar> cellVolumes(exec, nCells, 0.0);

    auto cvView = cellVolumes.view();
    auto ptsView = points.view();
    auto fnView = faceNodes.view();
    auto fcView = faceCentres.view();
    auto ccView = cellCentres.view();
    auto cfView = cellFaces.view();

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx c) {
            auto [cfStart, cfEnd] = cfView.bounds(c);
            scalar vol = 0.0;
            Vec3 cc = ccView[c];
            for (localIdx i = cfStart; i < cfEnd; ++i)
            {
                localIdx f = cfView.values[i];
                auto [fnStart, fnEnd] = fnView.bounds(f);
                Vec3 fc = fcView[f];
                localIdx nNodes = fnEnd - fnStart;
                for (localIdx n = 0; n < nNodes; ++n)
                {
                    localIdx curr = fnStart + n;
                    localIdx next = fnStart + ((n + 1) % nNodes);
                    vol += tetVolume(
                        cc, fc, ptsView[fnView.values[curr]], ptsView[fnView.values[next]]
                    );
                }
            }
            cvView[c] = vol;
        },
        "computeCellVolumes"
    );

    return cellVolumes;
}


MeshGeometry computeGeometry(
    const Executor& exec,
    const Vector<Vec3>& points,
    const Vector<localIdx>& faceOwner,
    const Vector<localIdx>& faceNeighbour,
    SegmentedVector<localIdx, localIdx>& faceNodes,
    localIdx nInternalFaces,
    localIdx nCells
)
{
    auto faceCentres = computeFaceCentres(exec, points, faceNodes);
    auto faceAreas = computeFaceAreas(exec, points, faceNodes, faceCentres);
    auto magFaceAreasVec = computeMagFaceAreas(exec, faceAreas);

    auto cellFaces = buildCellToFaceMapping(exec, faceOwner, faceNeighbour, nInternalFaces, nCells);
    auto cellCentres = computeCellCentres(exec, faceCentres, cellFaces, nCells);
    auto cellVolumes =
        computeCellVolumes(exec, points, faceNodes, faceCentres, cellCentres, cellFaces, nCells);

    return MeshGeometry {
        std::move(cellVolumes),
        std::move(cellCentres),
        std::move(faceAreas),
        std::move(faceCentres),
        std::move(magFaceAreasVec)
    };
}


MeshGeometry
computeGeometry(const std::vector<Vec3>& points, const FaceTopology& topo, localIdx nCells)
{
    SerialExecutor exec;

    // Convert points to NeoN Vector
    Vector<Vec3> devicePoints(exec, points);

    // faceOwner and faceNeighbour are already NeoN types in the new FaceTopology.
    // faceNodes needs a non-const copy since computeGeometry takes SegmentedVector&.
    auto faceNodesCopy = topo.faceNodes;

    return computeGeometry(
        exec,
        devicePoints,
        topo.faceOwner,
        topo.faceNeighbour,
        faceNodesCopy,
        topo.nInternalFaces,
        nCells
    );
}


} // namespace NeoN::io
