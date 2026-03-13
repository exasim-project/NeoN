// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp" // for Vec3


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
    return createUniform3DGrid(exec, nCells, 1, 1);
}

UnstructuredMesh
createUniform2DGrid(const Executor exec, localIdx nx, localIdx ny, scalar Lx, scalar Ly)
{
    return createUniform3DGrid(exec, nx, ny, 1, Lx, Ly, 1.0);
}

UnstructuredMesh createUniform3DGrid(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar Lx, scalar Ly, scalar Lz
)
{
    detail::GridParams p {
        nx,
        ny,
        nz,
        Lx / static_cast<scalar>(nx),
        Ly / static_cast<scalar>(ny),
        Lz / static_cast<scalar>(nz),
        Lx,
        Ly,
        Lz
    };

    auto pts = detail::generatePoints(p);
    auto [vols, centres] = detail::generateCellData(p);
    auto faces = detail::generateInternalFaces(p);
    auto [boundaryMesh, nBoundary] = detail::generateBoundaryData(exec, p, centres, faces);

    const localIdx nCells = nx * ny * nz;
    const localIdx nFaces = faces.nInternal + nBoundary;

    auto faceNodesVec = detail::buildFaceNodes(p, nFaces);

    UnstructuredMesh mesh(
        vectorVector(exec, pts),
        scalarVector(exec, vols),
        vectorVector(exec, centres),
        {exec, faces.areas},
        {exec, faces.centres},
        {exec, faces.magnitudes},
        {exec, faces.owner},
        labelVector(exec, faces.neighbour),
        nCells,
        faces.nInternal,
        nBoundary,
        6,
        nFaces,
        boundaryMesh
    );

    detail::storeFaceNodesInStencilDB(mesh, faceNodesVec);

    return mesh;
}
} // namespace NeoN
