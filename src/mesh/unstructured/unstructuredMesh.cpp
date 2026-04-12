// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/uniformMeshDataGenerator.hpp"

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
    BoundaryMesh boundaryMesh
)
    : exec_(exec), points_(points), cellVolumes_(cellVolumes), cellCentres_(cellCentres),
      faceAreas_(faceAreas), faceCentres_(faceCentres), magFaceAreas_(magFaceAreas),
      faceOwner_(faceOwner), faceNeighbour_(faceNeighbour), nCells_(cellVolumes.size()),
      nInternalFaces_(faceNeighbour.size()), boundaryMesh_(boundaryMesh), stencilDataBase_()
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

localIdx UnstructuredMesh::nBoundaryFaces() const { return boundaryMesh_.nBoundaryFaces(); }

localIdx UnstructuredMesh::nProcBoundaryFaces() const { return boundaryMesh_.nProcBoundaryFaces(); }

localIdx UnstructuredMesh::nBoundaries() const { return boundaryMesh_.nBoundaries(); }

localIdx UnstructuredMesh::nTotalFaces() const
{
    return nInternalFaces() + nBoundaryFaces() + nProcBoundaryFaces();
}

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
        {0, 1, 2, 3, 4},              // offset
        0,                            // number of proc boundary patches
        {}                            // neighbourRank
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
        boundaryMesh
    );
}

UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells, scalar Lx)
{
    return create3DUniformMesh(exec, nCells, 1, 1, Lx, 1.0, 1.0);
}

UnstructuredMesh
create2DUniformMesh(const Executor exec, localIdx nx, localIdx ny, scalar Lx, scalar Ly)
{
    return create3DUniformMesh(exec, nx, ny, 1, Lx, Ly, 1.0);
}

UnstructuredMesh create3DUniformMesh(
    const Executor exec, localIdx nx, localIdx ny, localIdx nz, scalar Lx, scalar Ly, scalar Lz
)
{
    // Validate input parameters
    NF_ASSERT(nx > 0 && ny > 0 && nz > 0, "Number of cells in each direction must be positive");
    NF_ASSERT(Lx > 0 && Ly > 0 && Lz > 0, "Domain lengths must be positive");

    // Hold the mesh parameters
    detail::MeshParams p {nx, ny, nz, Lx, Ly, Lz};

    const auto points = detail::generatePoints(p);
    const auto [cellVolumes, cellCentres] = detail::generateCellData(p);

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
        exec, dim, p, cellCentres, nInternalFaces, nBoundaryFaces, offset, faces
    );

    // Note: With the localIdx type (int32_t), the limit is 2 x 10^9 cells
    const localIdx nCells = nx * ny * nz;

    UnstructuredMesh mesh(
        vectorVector(exec, std::move(points)),
        scalarVector(exec, std::move(cellVolumes)),
        vectorVector(exec, std::move(cellCentres)),
        {exec, std::move(faces.areas)},
        {exec, std::move(faces.centres)},
        {exec, std::move(faces.magnitudes)},
        {exec, std::move(faces.owner)},
        labelVector(exec, std::move(faces.neighbour)),
        nCells,
        nInternalFaces,
        nBoundaryFaces,
        offset.size() - 1, // nBoundaries
        nFaces,
        std::move(boundaryMesh)
    );

    mesh.stencilDB().insert(std::string("stencilPatchNames"), patchNames);

    return mesh;
}

/* @brief helper to create a part of a global 1D mesh
 *
 */
std::pair<UnstructuredMesh, CommunicationPattern>
create1DUniformMeshPart(const Executor exec, const localIdx nCells, mpi::Environment mpiEnviron)
{
    Vec3 leftBoundary {static_cast<scalar>(mpiEnviron.rank()) / mpiEnviron.sizeRank(), 0.0, 0.0};
    Vec3 rightBoundary {
        static_cast<scalar>(mpiEnviron.rank() + 1) / mpiEnviron.sizeRank(), 0.0, 0.0
    };

    localIdx nProcBoundaryFaces = 2;
    if (mpiEnviron.rank() == 0 || mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        nProcBoundaryFaces = 1;
    }

    // regular boundary first, processor boundary follow
    auto faceCellVec = std::vector<localIdx>();
    auto boundaryWeights = std::vector<scalar>();
    if (mpiEnviron.rank() != 0 && mpiEnviron.rank() != mpiEnviron.sizeRank() - 1)
    {
        faceCellVec.push_back(0);
        faceCellVec.push_back(nCells - 1);
        boundaryWeights.push_back(-1.0);
        boundaryWeights.push_back(1.0);
    }
    // first rank
    if (mpiEnviron.rank() == 0)
    {
        faceCellVec.push_back(0);
        faceCellVec.push_back(nCells - 1);
        boundaryWeights.push_back(-1.0);
        boundaryWeights.push_back(1.0);
    }
    // last rank
    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        faceCellVec.push_back(nCells - 1);
        faceCellVec.push_back(0);
        boundaryWeights.push_back(1.0);
        boundaryWeights.push_back(-1.0);
    }

    labelVector faceCells(exec, faceCellVec);

    auto tmp = create1DUniformMesh(exec, nCells, leftBoundary, rightBoundary);
    BoundaryMesh boundaryMesh(
        exec,
        faceCells,
        {exec, {leftBoundary, rightBoundary}}, // cf
        tmp.boundaryMesh().cn(),               // cn
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}
        },                  // sf FIXME the order of the rest is potentially wrong
        {exec, {1.0, 1.0}}, // magSf
        {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}}, // nf
        tmp.boundaryMesh().delta(),                  // deltaCoeffs --> mag(1 / delta)
        {exec, boundaryWeights},                     // weights
        tmp.boundaryMesh().deltaCoeffs(),            // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2},                                   // offset
        nProcBoundaryFaces,                          // number of proc boundary patches
        {}                                           // neighbourRank
    );

    auto sendCounts = std::vector<int>(mpiEnviron.sizeRank() + 1);
    auto recvIdx = std::vector<int>();
    sendCounts[sendCounts.size() - 1] = 1;
    // middle
    // [0 1 2 3], [4 5 6 7] [8 9 10 11]
    if (mpiEnviron.rank() != 0 && mpiEnviron.rank() != mpiEnviron.sizeRank() - 1)
    {
        sendCounts[sendCounts.size() - 1] = 2;
        sendCounts[mpiEnviron.rank() - 1] = 1;
        sendCounts[mpiEnviron.rank() + 1] = 1;
        recvIdx.push_back(nCells - 1);
        recvIdx.push_back((mpiEnviron.rank() + 1) * nCells);
    }
    // first rank
    if (mpiEnviron.rank() == 0)
    {
        sendCounts[1] = 1;
        recvIdx.push_back(nCells);
    }
    // last rank
    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        sendCounts[mpiEnviron.rank() - 1] = 1;
        recvIdx.push_back(mpiEnviron.rank() * nCells - 1);
    }

    std::vector<localIdx> boundaryMapVector;

    // NOTE on rank2 the face centres [-1] and [-2] needs to be switched
    // since proc boundaries come first
    auto faceCentresH = tmp.faceCentres().copyToHost();
    auto faceAreasH = tmp.faceAreas().copyToHost();
    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        auto localCells = tmp.nCells();
        auto tmpValue = faceCentresH.view()[localCells];
        faceCentresH.view()[localCells] = faceCentresH.view()[localCells - 1];
        faceCentresH.view()[localCells - 1] = tmpValue;
        faceAreasH.view()[localCells] = {-1.0, 0.0, 0.0};
        faceAreasH.view()[localCells - 1] = {1.0, 0.0, 0.0};
    }

    // Note on rank the proc boundary needs to be turned

    return {
        UnstructuredMesh(
            tmp.points(),
            tmp.cellVolumes(),
            tmp.cellCentres(),
            faceAreasH.copyToExecutor(exec),
            faceCentresH.copyToExecutor(exec),
            tmp.magFaceAreas(),
            tmp.faceOwner(),
            tmp.faceNeighbour(),
            boundaryMesh
        ),
        // FIXME remove creation of comm pattern here. And make it a member of uniform mesh
        CommunicationPattern(sendCounts, recvIdx, boundaryMapVector, mpiEnviron)
    };

    // FIXME doesnt work on GPU
    // if (mpiEnviron.rank() != 0)
    // {
    //     ret.boundaryMesh().nf().view()[0] = {-1.0, 0.0, 0.0};
    //     ret.boundaryMesh().sf().view()[0] = {-1.0, 0.0, 0.0};
    // }

    // return ret;
}

} // namespace NeoN
