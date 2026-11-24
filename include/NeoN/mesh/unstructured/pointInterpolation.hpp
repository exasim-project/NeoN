// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/mesh/unstructured/trackingMesh.hpp"

namespace NeoN
{

/** @brief Barycentric coordinates w of x in the tetrahedron (a, b, c, d).
 *
 * GPU callable. x = w[0] a + w[1] b + w[2] c + w[3] d, with the w summing to 1.
 *
 * @return the determinant of the tetrahedron edges (6 x its signed volume). If it is
 *         zero (degenerate tet), w = (1, 0, 0, 0).
 */
KOKKOS_INLINE_FUNCTION
scalar pointToBarycentric(
    const Vec3& x, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, scalar w[4]
)
{
    const Vec3 e1 = b - a, e2 = c - a, e3 = d - a, r = x - a;
    const scalar det = (e1 ^ e2) & e3;
    if (det == scalar(0))
    {
        w[0] = scalar(1);
        w[1] = w[2] = w[3] = scalar(0);
        return det;
    }
    w[1] = ((r ^ e2) & e3) / det;
    w[2] = ((e1 ^ r) & e3) / det;
    w[3] = ((e1 ^ e2) & r) / det;
    w[0] = scalar(1) - w[1] - w[2] - w[3];
    return det;
}

/** @brief Interpolate a field at x in cell c from cell and point values, as OpenFOAM's
 * interpolationCellPoint.
 *
 * Finds the tet (cell centre, face base point, two consecutive face points) containing
 * x, as OpenFOAM cellPointWeight::findTetrahedron, and blends the cell value with the
 * three point values by the barycentric weights. If no tet contains x, the tet with the
 * largest smallest weight is used. If the cell has no valid tet, the cell value is
 * returned.
 *
 * @param m           tracking mesh view
 * @param cellValues  field value of every cell
 * @param pointValues field value of every mesh point (e.g. from volPointInterpolation)
 * @param c           cell containing x
 * @param x           position
 */
template<typename ValueType>
KOKKOS_INLINE_FUNCTION ValueType interpolateCellPoint(
    const TrackingMeshView& m,
    const View<const ValueType>& cellValues,
    const View<const ValueType>& pointValues,
    const label c,
    const Vec3& x
)
{
    constexpr scalar tol = scalar(1e-15);
    const Vec3 cc = m.cellCenters[c];
    scalar best[4] = {scalar(1), scalar(0), scalar(0), scalar(0)};
    label bestPts[3] = {-1, -1, -1};
    scalar bestMin = scalar(-1e300);
    bool found = false;
    const auto [faceStart, faceEnd] = m.cellFaces.bounds(c);
    for (localIdx k = faceStart; k < faceEnd && !found; k++)
    {
        const label face = m.cellFaces.values[k];
        const auto [pointStart, pointEnd] = m.facePoints.bounds(face);
        const label p0 = label(pointStart);
        const label n = label(pointEnd) - p0;
        const label base = m.faceBasePt[face];
        for (label tetPt = 1; tetPt < n - 1; tetPt++)
        {
            // OpenFOAM tetIndices::faceTriIs (flipped for the neighbour cell)
            label facePt = (tetPt + base) % n;
            label otherPt = (facePt + 1) % n;
            if (m.faceOwners[face] != c)
            {
                const label t = facePt;
                facePt = otherPt;
                otherPt = t;
            }
            const label ia = m.facePoints.values[p0 + base];
            const label ib = m.facePoints.values[p0 + facePt];
            const label ic = m.facePoints.values[p0 + otherPt];
            scalar w[4];
            const scalar det =
                pointToBarycentric(x, cc, m.points[ia], m.points[ib], m.points[ic], w);
            if (Kokkos::fabs(det / m.cellVolumes[c]) <= tol)
            {
                continue;
            }
            const scalar wMin = Kokkos::fmin(Kokkos::fmin(w[0], w[1]), Kokkos::fmin(w[2], w[3]));
            if (wMin > bestMin)
            {
                bestMin = wMin;
                for (int j = 0; j < 4; j++)
                {
                    best[j] = w[j];
                }
                bestPts[0] = ia;
                bestPts[1] = ib;
                bestPts[2] = ic;
            }
            if (wMin + tol > scalar(0))
            {
                found = true;
                break;
            }
        }
    }
    if (bestPts[0] < 0)
    {
        return cellValues[c];
    }
    return best[0] * cellValues[c] + best[1] * pointValues[bestPts[0]]
         + best[2] * pointValues[bestPts[1]] + best[3] * pointValues[bestPts[2]];
}

} // namespace NeoN
