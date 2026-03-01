// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/geometry/cellGeometry.hpp"

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include <cmath>


namespace NeoN::io
{

namespace
{

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


} // namespace NeoN::io
