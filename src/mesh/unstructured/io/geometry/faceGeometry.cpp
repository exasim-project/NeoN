// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/geometry/faceGeometry.hpp"

#include "NeoN/core/parallelAlgorithms.hpp"

#include <cmath>


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


} // namespace NeoN::io
