// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp" // for Vec3
#include <cmath>


namespace NeoN
{
UnstructuredMesh::UnstructuredMesh(
    Executor exec,
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCentres,
    vectorVector faceAreas,
    vectorVector faceCentres,
    scalarVector magFaceAreas,
    labelVector faceOwner,
    labelVector faceNeighbour,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    localIdx nBoundaries,
    localIdx nFaces,
    BoundaryMesh boundaryMesh
)
    : exec_(exec), points_(points), cellVolumes_(cellVolumes), cellCentres_(cellCentres),
      faceAreas_(faceAreas), faceCentres_(faceCentres), magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner), faceNeighbour_(faceNeighbour), nCells_(nCells),
      nInternalFaces_(nInternalFaces), nBoundaryFaces_(nBoundaryFaces), nBoundaries_(nBoundaries),
      nFaces_(nFaces), boundaryMesh_(boundaryMesh), stencilDataBase_()
{}

UnstructuredMesh::UnstructuredMesh(
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCentres,
    vectorVector faceAreas,
    vectorVector faceCentres,
    scalarVector magFaceAreas,
    labelVector faceOwner,
    labelVector faceNeighbour,
    localIdx nCells,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    localIdx nBoundaries,
    localIdx nFaces,
    BoundaryMesh boundaryMesh
)
    : UnstructuredMesh(
        faceOwner.exec(),
        points,
        cellVolumes,
        cellCentres,
        faceAreas,
        faceCentres,
        magFaceAreas,
        faceOwner,
        faceNeighbour,
        nCells,
        nInternalFaces,
        nBoundaryFaces,
        nBoundaries,
        nFaces,
        boundaryMesh
    )
{}


const vectorVector& UnstructuredMesh::points() const { return points_; }

vectorVector& UnstructuredMesh::points() { return points_; }

const scalarVector& UnstructuredMesh::cellVolumes() const { return cellVolumes_; }

scalarVector& UnstructuredMesh::cellVolumes() { return cellVolumes_; }

const vectorVector& UnstructuredMesh::cellCentres() const { return cellCentres_; }

vectorVector& UnstructuredMesh::cellCentres() { return cellCentres_; }

const vectorVector& UnstructuredMesh::faceCentres() const { return faceCentres_; }

vectorVector& UnstructuredMesh::faceCentres() { return faceCentres_; }

const vectorVector& UnstructuredMesh::faceAreas() const { return faceAreas_; }

vectorVector& UnstructuredMesh::faceAreas() { return faceAreas_; }

const scalarVector& UnstructuredMesh::magFaceAreas() const { return magFaceAreas_; }

scalarVector& UnstructuredMesh::magFaceAreas() { return magFaceAreas_; }

const labelVector& UnstructuredMesh::faceOwner() const { return faceOwner_; }

labelVector& UnstructuredMesh::faceOwner() { return faceOwner_; }

const labelVector& UnstructuredMesh::faceNeighbour() const { return faceNeighbour_; }

labelVector& UnstructuredMesh::faceNeighbour() { return faceNeighbour_; }

localIdx UnstructuredMesh::nCells() const { return nCells_; }

localIdx UnstructuredMesh::nInternalFaces() const { return nInternalFaces_; }

localIdx UnstructuredMesh::nBoundaryFaces() const { return nBoundaryFaces_; }

localIdx UnstructuredMesh::nBoundaries() const { return nBoundaries_; }

localIdx UnstructuredMesh::nFaces() const { return nFaces_; }

const BoundaryMesh& UnstructuredMesh::boundaryMesh() const { return boundaryMesh_; }

Dictionary& UnstructuredMesh::stencilDB() const { return stencilDataBase_; }

const Executor& UnstructuredMesh::exec() const { return exec_; }

UnstructuredMesh createSingleCellMesh(const Executor exec)
{
    // a 2D mesh in 3D space with left, right, top, bottom boundary faces
    // with the centre at (0.5, 0.5, 0.0)
    // left, top, right, bottom faces
    // and four boundaries one left, right, top, bottom

    vectorVector faceAreasVec3s(exec, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, -1, 0}});
    vectorVector faceCentresVec3s(
        exec, {{0.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {1.0, 0.5, 0.0}, {0.5, 0.0, 0.0}}
    );
    scalarVector magFaceAreas(exec, {1, 1, 1, 1});

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, 0, 0, 0}},                                                           // faceCells
        faceCentresVec3s,                                                               // cf
        faceAreasVec3s,                                                                 // cn,
        faceAreasVec3s,                                                                 // sf,
        magFaceAreas,                                                                   // magSf,
        faceAreasVec3s,                                                                 // nf,
        {exec, {{-0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}}}, // delta
        {exec, {1, 1, 1, 1}},                                                           // weights
        {exec, {2.0, 2.0, 2.0, 2.0}}, // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2, 3, 4}               // offset
    );
    return UnstructuredMesh(
        {exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}, // points,
        {exec, 1, 1.0},                                       // cellVolumes
        {exec, {{0.5, 0.5, 0.0}}},                            // cellCentres
        faceAreasVec3s,
        faceCentresVec3s,
        magFaceAreas,
        {exec, {0, 0, 0, 0}}, // faceOwner
        {exec, {}},           // faceNeighbour,
        1,                    // nCells
        0,                    // nInternalFaces,
        4,                    // nBoundaryFaces,
        4,                    // nBoundaries,
        4,                    // nFaces,
        boundaryMesh
    );
}

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
    auto faceNodesPtr =
        std::make_shared<std::vector<std::vector<localIdx>>>(static_cast<size_t>(nFaces));
    auto& faceNodesVec = *faceNodesPtr;

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

    mesh.stencilDB().insert(std::string("io::faceNodes"), faceNodesPtr);

    auto patchNames = std::make_shared<std::vector<std::string>>(
        std::vector<std::string> {"left", "right", "bottom", "top", "front", "back"}
    );
    mesh.stencilDB().insert(std::string("io::patchNames"), patchNames);

    return mesh;
}

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
    auto faceNodesPtr =
        std::make_shared<std::vector<std::vector<localIdx>>>(static_cast<size_t>(nFaces));
    auto& faceNodesVec = *faceNodesPtr;

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

    mesh.stencilDB().insert(std::string("io::faceNodes"), faceNodesPtr);

    auto patchNames = std::make_shared<std::vector<std::string>>(
        std::vector<std::string> {"left", "right", "bottom", "top", "front", "back"}
    );
    mesh.stencilDB().insert(std::string("io::patchNames"), patchNames);

    return mesh;
}

} // namespace NeoN
