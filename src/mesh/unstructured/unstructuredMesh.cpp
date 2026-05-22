// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/uniformMeshDataGenerator.hpp"

#include "NeoN/core/primitives/vec3.hpp" // for Vec3


namespace NeoN
{

localIdx computeGlobalOffset(const BoundaryMesh& boundaryMesh, localIdx localNCells)
{
    // NOTE not yet implemented, will be added via separate PR
    // if (!boundaryMesh.isDistributed())
    // {
    //     return 0;
    // }
    // mpi::Environment mpiEnviron;
    // const auto nRanks = mpiEnviron.sizeRank();
    // const auto myRank = mpiEnviron.rank();
    // auto allNCells = std::vector<int>(nRanks);
    // MPI_Allgather(&localNCells, 1, MPI_INT, allNCells.data(), 1, MPI_INT, mpiEnviron.comm());
    // std::vector<localIdx> globalOffset(nRanks + 1, 0);
    // for (int i = 0; i < nRanks; i++)
    // {
    //     globalOffset[i + 1] = globalOffset[i] + allNCells[i];
    // }
    // return globalOffset[myRank];
    return 0;
}


UnstructuredMesh::UnstructuredMesh(
    Executor exec,
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCenters,
    vectorVector faceNormals,
    vectorVector faceCenters,
    scalarVector faceAreas,
    labelVector faceOwners,
    labelVector faceNeighbors,
    BoundaryMesh boundaryMesh
)
    : exec_(exec), points_(points), cellVolumes_(cellVolumes), cellCenters_(cellCenters),
      faceNormals_(faceNormals), faceCenters_(faceCenters), faceAreas_(faceAreas),
      faceOwners_(faceOwners), faceNeighbors_(faceNeighbors), nCells_(cellVolumes.size()),
      nInternalFaces_(faceNeighbors.size()), boundaryMesh_(boundaryMesh),
      globalOffset_(computeGlobalOffset(boundaryMesh, cellVolumes.size())), stencilDataBase_()
{}

UnstructuredMesh::UnstructuredMesh(
    vectorVector points,
    scalarVector cellVolumes,
    vectorVector cellCenters,
    vectorVector faceNormals,
    vectorVector faceCenters,
    scalarVector faceAreas,
    labelVector faceOwners,
    labelVector faceNeighbors,
    BoundaryMesh boundaryMesh
)
    : UnstructuredMesh(
        faceOwners.exec(),
        points,
        cellVolumes,
        cellCenters,
        faceNormals,
        faceCenters,
        faceAreas,
        faceOwners,
        faceNeighbors,
        boundaryMesh
    )
{}


const vectorVector& UnstructuredMesh::points() const { return points_; }

vectorVector& UnstructuredMesh::points() { return points_; }

const scalarVector& UnstructuredMesh::cellVolumes() const { return cellVolumes_; }

scalarVector& UnstructuredMesh::cellVolumes() { return cellVolumes_; }

const vectorVector& UnstructuredMesh::cellCenters() const { return cellCenters_; }

vectorVector& UnstructuredMesh::cellCenters() { return cellCenters_; }

const vectorVector& UnstructuredMesh::faceCenters() const { return faceCenters_; }

vectorVector& UnstructuredMesh::faceCenters() { return faceCenters_; }

const vectorVector& UnstructuredMesh::faceNormals() const { return faceNormals_; }

vectorVector& UnstructuredMesh::faceNormals() { return faceNormals_; }

const scalarVector& UnstructuredMesh::faceAreas() const { return faceAreas_; }

scalarVector& UnstructuredMesh::faceAreas() { return faceAreas_; }

const labelVector& UnstructuredMesh::faceOwners() const { return faceOwners_; }

labelVector& UnstructuredMesh::faceOwners() { return faceOwners_; }

const labelVector& UnstructuredMesh::faceNeighbors() const { return faceNeighbors_; }

labelVector& UnstructuredMesh::faceNeighbors() { return faceNeighbors_; }

localIdx UnstructuredMesh::nCells() const { return nCells_; }

localIdx UnstructuredMesh::nInternalFaces() const { return nInternalFaces_; }

localIdx UnstructuredMesh::nBoundaryFaces() const { return boundaryMesh_.nBoundaryFaces(); }

localIdx UnstructuredMesh::nProcBoundaryFaces() const { return boundaryMesh_.nProcBoundaryFaces(); }

localIdx UnstructuredMesh::nBoundaries() const { return boundaryMesh_.nBoundaries(); }

localIdx UnstructuredMesh::nTotalFaces() const
{
    return nInternalFaces() + nBoundaryFaces() + nProcBoundaryFaces();
}

localIdx UnstructuredMesh::globalOffset() const { return globalOffset_; }

const BoundaryMesh& UnstructuredMesh::boundaryMesh() const { return boundaryMesh_; }

BoundaryMesh& UnstructuredMesh::boundaryMesh() { return boundaryMesh_; }

Dictionary& UnstructuredMesh::stencilDB() const { return stencilDataBase_; }

const Executor& UnstructuredMesh::exec() const { return exec_; }

UnstructuredMesh createSingleCellMesh(const Executor exec)
{
    // a 2D mesh in 3D space with left, right, top, bottom boundary faces
    // with the centre at (0.5, 0.5, 0.0)
    // left, top, right, bottom faces
    // and four boundaries one left, right, top, bottom

    vectorVector faceAreasVec3s(exec, {{-1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, -1, 0}});
    vectorVector faceCentersVec3s(
        exec, {{0.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {1.0, 0.5, 0.0}, {0.5, 0.0, 0.0}}
    );
    scalarVector faceAreas(exec, {1, 1, 1, 1});

    BoundaryMesh boundaryMesh(
        exec,
        {exec, {0, 0, 0, 0}},                                                           // faceCells
        faceCentersVec3s,                                                               // cf
        faceAreasVec3s,                                                                 // cn,
        faceAreasVec3s,                                                                 // sf,
        faceAreas,                                                                      // magSf,
        faceAreasVec3s,                                                                 // nf,
        {exec, {{-0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}}}, // delta
        {exec, {1, 1, 1, 1}},                                                           // weights
        {exec, {2.0, 2.0, 2.0, 2.0}}, // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2, 3, 4},              // offset
        0,                            // number of proc boundary patches
        {}                            // neighbourRank
    );
    return UnstructuredMesh(
        {exec, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}, // points,
        {exec, 1, 1.0},                                       // cellVolumes
        {exec, {{0.5, 0.5, 0.0}}},                            // cellCenters
        faceAreasVec3s,
        faceCentersVec3s,
        faceAreas,
        {exec, {0, 0, 0, 0}}, // faceOwners
        {exec, {}},           // faceNeighbors
        boundaryMesh
    );
}

UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells, scalar lx)
{
    return create3DUniformMesh(exec, nCells, 1, 1, lx, 1.0, 1.0);
}

UnstructuredMesh
create2DUniformMesh(const Executor exec, localIdx nx, localIdx ny, scalar lx, scalar ly)
{
    return create3DUniformMesh(exec, nx, ny, 1, lx, ly, 1.0);
}

UnstructuredMesh create3DUniformMesh(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar lx, scalar ly, scalar lz
)
{
    // Validate input parameters
    NF_ASSERT(nx > 0 && ny > 0 && nz > 0, "Number of cells in each direction must be positive");
    NF_ASSERT(lx > 0 && ly > 0 && lz > 0, "Domain lengths must be positive");

    // Hold the mesh parameters
    detail::MeshParams p {nx, ny, nz, lx, ly, lz};

    const auto points = detail::generatePoints(p);
    const auto [cellVolumes, cellCenters] = detail::generateCellData(p);

    // Judge the dimension based on the input parameters
    int dim = 0;
    if (ny == 1 && nz == 1)
    {
        dim = 1;
    }
    else if (nz == 1)
    {
        dim = 2;
    }
    else
    {
        dim = 3;
    }

    // Compute the number of internal faces and boundary faces based on the mesh parameters
    const localIdx nXInternalFaces = (p.nx - 1) * p.ny * p.nz;
    const localIdx nYInternalFaces = p.nx * (p.ny - 1) * p.nz;
    const localIdx nZInternalFaces = p.nx * p.ny * (p.nz - 1);
    const localIdx nInternalFaces = nXInternalFaces + nYInternalFaces + nZInternalFaces;

    const localIdx nBndLeft = p.ny * p.nz;
    const localIdx nBndRight = p.ny * p.nz;

    std::vector<localIdx> offset = {0, nBndLeft, nBndLeft + nBndRight};
    // std::vector<std::string> patchNames = {"xmin", "xmax"};
    auto patchNames =
        std::make_shared<std::vector<std::string>>(std::vector<std::string> {"xmin", "xmax"});

    // If the mesh is more than 1D, there are bottom and top boundary faces
    if (dim > 1)
    {
        const localIdx nBndBottom = p.nx * p.nz;
        const localIdx nBndTop = p.nx * p.nz;
        offset.push_back(offset.back() + nBndBottom);
        offset.push_back(offset.back() + nBndTop);
        patchNames->push_back("ymin");
        patchNames->push_back("ymax");
    }

    // If the mesh is more than 2D, there are front and back boundary faces
    if (dim > 2)
    {
        const localIdx nBndFront = p.nx * p.ny;
        const localIdx nBndBack = p.nx * p.ny;
        offset.push_back(offset.back() + nBndFront);
        offset.push_back(offset.back() + nBndBack);
        patchNames->push_back("zmin");
        patchNames->push_back("zmax");
    }

    const localIdx nBoundaryFaces = offset.back();
    const localIdx nFaces = nInternalFaces + nBoundaryFaces;

    auto faces = detail::generateInternalFaces(p, nInternalFaces, nFaces);
    auto boundaryMesh = detail::generateBoundaryData(
        exec, dim, p, cellCenters, nInternalFaces, nBoundaryFaces, offset, faces
    );

    // Note: With the localIdx type (int32_t), the limit is 2 x 10^9 cells
    const localIdx nCells = nx * ny * nz;

    UnstructuredMesh mesh(
        vectorVector(exec, std::move(points)),
        scalarVector(exec, std::move(cellVolumes)),
        vectorVector(exec, std::move(cellCenters)),
        {exec, std::move(faces.areas)},
        {exec, std::move(faces.centers)},
        {exec, std::move(faces.magnitudes)},
        {exec, std::move(faces.owner)},
        labelVector(exec, std::move(faces.neighbour)),
        std::move(boundaryMesh)
    );

    mesh.stencilDB().insert(std::string("stencilPatchNames"), patchNames);

    return mesh;
}


} // namespace NeoN
