// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp"

namespace NeoN
{

UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells)
{
    const Vec3 leftBoundary = {0.0, 0.0, 0.0};
    const Vec3 rightBoundary = {1.0, 0.0, 0.0};
    scalar meshSpacing = (rightBoundary[0] - leftBoundary[0]) / static_cast<scalar>(nCells);
    auto hostExec = SerialExecutor {};
    vectorVector meshPointsHost(hostExec, nCells + 1, {0.0, 0.0, 0.0});
    auto meshPointsHostView = meshPointsHost.view();
    meshPointsHostView[nCells - 1] = leftBoundary;
    meshPointsHostView[nCells] = rightBoundary;
    auto meshPoints = meshPointsHost.copyToExecutor(exec);

    // loop over internal mesh points
    auto meshPointsView = meshPoints.view();
    auto leftBoundaryX = leftBoundary[0];
    parallelFor(
        exec,
        {0, nCells - 1},
        NEON_LAMBDA(const localIdx i) {
            meshPointsView[i][0] = leftBoundaryX + static_cast<scalar>(i + 1) * meshSpacing;
        },
        "computeMeshPoints"
    );

    scalarVector cellVolumes(exec, nCells, meshSpacing);

    vectorVector cellCenters(exec, nCells, {0.0, 0.0, 0.0});
    auto cellCentersView = cellCenters.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i) {
            cellCentersView[i][0] = 0.5 * meshSpacing + meshSpacing * static_cast<scalar>(i);
        },
        "computeCellCenters"
    );


    vectorVector faceAreasHost(hostExec, nCells + 1, {1.0, 0.0, 0.0});
    auto faceAreasHostView = faceAreasHost.view();
    faceAreasHostView[nCells - 1] = {-1.0, 0.0, 0.0}; // left boundary face
    auto faceAreas = faceAreasHost.copyToExecutor(exec);

    vectorVector faceCenters(exec, meshPoints);
    scalarVector magFaceAreas(exec, nCells + 1, 1.0);

    labelVector faceOwnerHost(hostExec, nCells + 1);
    labelVector faceNeighbor(exec, nCells - 1);
    auto faceOwnerHostView = faceOwnerHost.view();
    faceOwnerHostView[nCells - 1] = 0;                          // left boundary face
    faceOwnerHostView[nCells] = static_cast<label>(nCells) - 1; // right boundary face
    auto faceOwner = faceOwnerHost.copyToExecutor(exec);

    // loop over internal faces
    auto faceOwnerView = faceOwner.view();
    auto faceNeighborView = faceNeighbor.view();
    parallelFor(
        exec,
        {0, nCells - 1},
        NEON_LAMBDA(const localIdx i) {
            faceOwnerView[i] = i;
            faceNeighborView[i] = i + 1;
        },
        "computeFaceOwnerAndNeighbors"
    );

    vectorVector deltaHost(hostExec, 2);
    auto deltaHostView = deltaHost.view();
    auto cellCentersHost = cellCenters.copyToHost();
    auto cellCentersHostView = cellCentersHost.view();
    deltaHostView[0] = {leftBoundary[0] - cellCentersHostView[0][0], 0.0, 0.0};
    deltaHostView[1] = {rightBoundary[0] - cellCentersHostView[nCells - 1][0], 0.0, 0.0};
    auto delta = deltaHost.copyToExecutor(exec);

    scalarVector deltaCoeffsHost(hostExec, 2);
    auto deltaCoeffsHostView = deltaCoeffsHost.view();
    deltaCoeffsHostView[0] = 1 / mag(deltaHostView[0]);
    deltaCoeffsHostView[1] = 1 / mag(deltaHostView[1]);
    auto deltaCoeffs = deltaCoeffsHost.copyToExecutor(exec);

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, nCells - 1}},
        {exec, {leftBoundary, rightBoundary}},
        {exec, {cellCentersHostView[0], cellCentersHostView[nCells - 1]}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        {exec, {1.0, 1.0}},
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
        delta,
        {exec, {1.0, 1.0}},
        deltaCoeffs,
        {0, 1, 2}
    );

    return UnstructuredMesh(
        meshPoints,
        cellVolumes,
        cellCenters,
        faceAreas,
        faceCenters,
        magFaceAreas,
        faceOwner,
        faceNeighbor,
        nCells,
        nCells - 1,
        2,
        2,
        nCells + 1,
        boundaryMesh
    );
}

} // namespace NeoN
