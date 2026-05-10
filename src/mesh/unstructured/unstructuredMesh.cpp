// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <numeric>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/uniformMeshDataGenerator.hpp"

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/primitives/vec3.hpp" // for Vec3

#include "NeoN/core/mpi/operators.hpp"

namespace NeoN
{

localIdx computeGlobalOffset(const BoundaryMesh& boundaryMesh, localIdx localNCells)
{
    if (!boundaryMesh.isDistributed())
    {
        return 0;
    }
    mpi::Environment mpiEnviron;
    const auto nRanks = mpiEnviron.sizeRank();
    const auto myRank = mpiEnviron.rank();
    auto allNCells = std::vector<int>(nRanks);
    MPI_Allgather(&localNCells, 1, MPI_INT, allNCells.data(), 1, MPI_INT, mpiEnviron.comm());
    std::vector<localIdx> globalOffset(nRanks + 1, 0);
    for (int i = 0; i < nRanks; i++)
    {
        globalOffset[i + 1] = globalOffset[i] + allNCells[i];
    }
    return globalOffset[myRank];
}

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
      nInternalFaces_(faceNeighbour.size()), boundaryMesh_(boundaryMesh),
      globalOffset_(computeGlobalOffset(boundaryMesh, cellVolumes.size())), stencilDataBase_()
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
        std::move(boundaryMesh)
    );

    mesh.stencilDB().insert(std::string("stencilPatchNames"), patchNames);

    return mesh;
}

/* @brief helper to create a part of a global 1D mesh
 *
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells)
{
    // FIXME make it an argument again
    mpi::Environment mpiEnviron;
    scalar rightBoundary {static_cast<scalar>(1.0) / mpiEnviron.sizeRank()};

    localIdx nProcBoundaryFaces = 2;
    if (mpiEnviron.rank() == 0 || mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        nProcBoundaryFaces = 1;
    }

    // regular boundary first, processor boundary follow
    auto faceCellVec = std::vector<localIdx>();
    auto boundaryWeights = std::vector<scalar>();
    auto neighRanks = std::vector<localIdx>();
    if (mpiEnviron.rank() != 0 && mpiEnviron.rank() != mpiEnviron.sizeRank() - 1)
    {
        faceCellVec.push_back(0);
        faceCellVec.push_back(nCells - 1);
        boundaryWeights.push_back(-1.0);
        boundaryWeights.push_back(1.0);
        neighRanks.push_back(mpiEnviron.rank() - 1);
        neighRanks.push_back(mpiEnviron.rank() + 1);
    }
    // first rank
    if (mpiEnviron.rank() == 0)
    {
        faceCellVec.push_back(0);
        faceCellVec.push_back(nCells - 1);
        boundaryWeights.push_back(-1.0);
        boundaryWeights.push_back(1.0);
        neighRanks.push_back(1);
    }
    // last rank
    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        faceCellVec.push_back(nCells - 1);
        faceCellVec.push_back(0);
        boundaryWeights.push_back(1.0);
        boundaryWeights.push_back(-1.0);
        neighRanks.push_back(mpiEnviron.rank() - 1);
    }

    labelVector faceCells(exec, faceCellVec);

    // The boundaryMesh stores boundary patches in the order they appear in
    // faceCellVec ("regular boundary first, processor boundary follow"). On the
    // last rank that order is [xmax_regular, proc_left], so the entry at index 0
    // is the right-side face and the entry at index 1 is the left-side face —
    // the opposite of rank 0 / middle ranks. cf/sf/nf must reflect this so that
    // proc-face deltaCoeffs (which reads bm.cf()/bm.sf() in compressed indexing)
    // computes 1/cellWidth instead of 7/cellWidth on the last rank.
    std::vector<Vec3> bcCfVec {{0.0, 0.0, 0.0}, {rightBoundary, 0.0, 0.0}};
    std::vector<Vec3> bcSfVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Vec3> bcNfVec {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    if (mpiEnviron.rank() == mpiEnviron.sizeRank() - 1)
    {
        std::swap(bcCfVec[0], bcCfVec[1]);
        std::swap(bcSfVec[0], bcSfVec[1]);
        std::swap(bcNfVec[0], bcNfVec[1]);
    }

    auto tmp = create1DUniformMesh(exec, nCells, rightBoundary);
    BoundaryMesh boundaryMesh(
        exec,
        faceCells,
        {exec, bcCfVec},                  // cf
        tmp.boundaryMesh().cn(),          // cn
        {exec, bcSfVec},                  // sf
        {exec, {1.0, 1.0}},               // magSf
        {exec, bcNfVec},                  // nf
        tmp.boundaryMesh().delta(),       // deltaCoeffs --> mag(1 / delta)
        {exec, boundaryWeights},          // weights
        tmp.boundaryMesh().deltaCoeffs(), // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2},                        // offset
        nProcBoundaryFaces,               // number of proc boundary patches
        neighRanks                        // neighbourRank
    );
    /*
        auto tmp = create1DUniformMesh(exec, nCells, rightBoundary);
        BoundaryMesh boundaryMesh(
            exec,
            faceCells,
            {exec, {{0.0, 0.0, 0.0}, {rightBoundary, 0.0, 0.0}}}, // cf
            tmp.boundaryMesh().cn(),                              // cn
            {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}
            },                  // sf FIXME the order of the rest is potentially wrong
            {exec, {1.0, 1.0}}, // magSf
            {exec, {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}}, // nf
            tmp.boundaryMesh().delta(),                  // deltaCoeffs --> mag(1 / delta)
            {exec, boundaryWeights},                     // weights
            tmp.boundaryMesh().deltaCoeffs(),            // deltaCoeffs --> mag(1 / delta)
            {0, 1, 2},                                   // offset
            nProcBoundaryFaces,                          // number of proc boundary patches
            neighRanks                                   // neighbourRank
        );
    */
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
        tmp.points(),
        tmp.cellVolumes(),
        tmp.cellCentres(),
        faceAreasH.copyToExecutor(exec),
        faceCentresH.copyToExecutor(exec),
        tmp.magFaceAreas(),
        tmp.faceOwner(),
        tmp.faceNeighbour(),
        boundaryMesh
    };

    // FIXME doesnt work on GPU
    // if (mpiEnviron.rank() != 0)
    // {
    //     ret.boundaryMesh().nf().view()[0] = {-1.0, 0.0, 0.0};
    //     ret.boundaryMesh().sf().view()[0] = {-1.0, 0.0, 0.0};
    // }

    // return ret;
}

CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh)
{
    mpi::Environment mpiEnviron;
    // early return if not distributed
    if (!mesh.boundaryMesh().isDistributed())
    {
        return {};
    }

    const auto nCells = mesh.nCells();
    auto sendCounts = std::vector<int>(mpiEnviron.sizeRank() + 1, 0);

    const auto neighbourRanks = mesh.boundaryMesh().neighbourRank();
    const auto offsets = mesh.boundaryMesh().offset();
    const auto nInnerBoundaries =
        mesh.boundaryMesh().nBoundaries() - mesh.boundaryMesh().nProcBoundaryPatches();

    for (int i = 0; i < neighbourRanks.size(); i++)
    {
        auto targetRank = neighbourRanks[i];
        auto patchSize = offsets[nInnerBoundaries + i + 1] - offsets[nInnerBoundaries + i];
        sendCounts[targetRank] = patchSize;
        sendCounts[mpiEnviron.sizeRank()] += patchSize;
    }

    // Build send buffer in MESH-BOUNDARY order: each proc patch is appended in
    // the order it appears in the boundary mesh (the natural layout of
    // boundaryMesh().faceCells()).
    //
    // recvIdx is then naturally in MESH-BOUNDARY order on the receiving side,
    // which is what the consumer in setProcBoundarySparsityPattern expects:
    // procRowOffs[i] holds the local mesh-order owner of proc-face i, paired
    // with procColIdx[i] = recvIdx[i] = the global ghost cell across that
    // same proc-face i.
    //
    // The MPI Alltoallv displacement for rank r MUST therefore point at the
    // mesh-order patch targeting rank r, NOT at the running cumulative sum of
    // sendCounts (which only matches mesh-order on decompositions where
    // neighbourRanks is already ascending). The previous implementation used
    // the running-sum approach and shipped wrong data to wrong ranks on every
    // non-ascending decomposition — and even after the data arrived at the
    // right rank (in earlier sort-buffer-by-rank attempts) the recvIdx fell
    // out of sync with mesh-order procRowOffs and corrupted the sparsity
    // pattern. Doing the per-rank lookup here keeps both invariants intact.
    auto globalOffset = mesh.globalOffset();
    const auto faceCells = mesh.boundaryMesh().faceCells();
    auto faceCellsH = faceCells.copyToHost();

    auto buffer = std::vector<localIdx>();
    buffer.reserve(mesh.boundaryMesh().nProcBoundaryFaces());
    auto procStart = offsets[nInnerBoundaries];
    for (int i = 0; i < mesh.boundaryMesh().nProcBoundaryFaces(); i++)
    {
        buffer.push_back(faceCellsH.view()[i + procStart] + globalOffset);
    }

    // For each mesh-order proc patch i with target rank neighbourRanks[i], the
    // displacement into the proc-tail buffer for rank r is the patch start
    // offset relative to procStart. Ranks that don't communicate keep the
    // default 0 displacement (sendCounts[r] == 0 means no bytes are read).
    auto sdispl = std::vector<localIdx>(mpiEnviron.sizeRank(), 0);
    for (int i = 0; i < neighbourRanks.size(); i++)
    {
        const auto targetRank = neighbourRanks[i];
        sdispl[targetRank] = offsets[nInnerBoundaries + i] - procStart;
    }
    auto recvIdx = std::vector<localIdx>(buffer.size());

    // MPI-01 fix: derive per-rank recv counts by exchanging send counts.
    auto recvCounts = std::vector<int>(mpiEnviron.sizeRank(), 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, mpiEnviron.comm());
    auto rdispl = std::vector<int>(mpiEnviron.sizeRank(), 0);
    for (int r = 1; r < mpiEnviron.sizeRank(); ++r)
        rdispl[r] = rdispl[r - 1] + recvCounts[r - 1];
    int totalRecv = rdispl.back() + recvCounts.back();
    recvIdx.resize(static_cast<std::size_t>(totalRecv));

    MPI_Alltoallv(
        buffer.data(),
        sendCounts.data(),
        sdispl.data(),
        mpi::getType<localIdx>(),
        recvIdx.data(),
        recvCounts.data(),
        rdispl.data(),
        mpi::getType<localIdx>(),
        mpiEnviron.comm()
    );

    // FIXME seems unused
    std::vector<localIdx> boundaryMapVector;
    return CommunicationPattern(sendCounts, recvIdx, boundaryMapVector, mpiEnviron);
}

} // namespace NeoN
