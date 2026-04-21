// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

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

UnstructuredMesh create1DUniformMesh(
    const Executor exec, const localIdx nCells, Vec3 leftBoundary, Vec3 rightBoundary
)
{
    // const Vec3 leftBoundary = {0.0, 0.0, 0.0};
    // const Vec3 rightBoundary = {1.0, 0.0, 0.0};
    scalar meshSpacing = (rightBoundary[0] - leftBoundary[0]) / static_cast<scalar>(nCells);
    auto hostExec = SerialExecutor {};
    vectorVector meshPointsHost(hostExec, nCells + 1, leftBoundary);
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

    vectorVector cellCenters(exec, nCells, leftBoundary);
    auto cellCentersView = cellCenters.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx i) {
            cellCentersView[i][0] =
                0.5 * meshSpacing + leftBoundary[0] + meshSpacing * static_cast<scalar>(i);
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
        {exec, {1.0, 1.0}}, // weights
        deltaCoeffs,        // deltaCoeffs --> mag(1 / delta)
        {0, 1, 2},          // offset
        0,                  // number of proc boundary patches
        {}                  // neighbourRank
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
        boundaryMesh
    );
}

/* @brief helper to create a part of a global 1D mesh
 *
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells)
{
    // FIXME make it an argument again
    mpi::Environment mpiEnviron;
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
        neighRanks                                   // neighbourRank
    );

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

    // compute sendIdx
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

    // exchange sendIdx
    // TODO this is assumes that patches and indices are ordered
    // compute send displacements
    auto sdispl = std::vector<localIdx>(mpiEnviron.sizeRank(), 0);
    for (int i = 1; i < sdispl.size(); i++)
    {
        sdispl[i] = sdispl[i - 1] + sendCounts[i - 1];
    }
    auto recvIdx = std::vector<localIdx>(buffer.size());

    MPI_Alltoallv(
        buffer.data(),
        sendCounts.data(),
        sdispl.data(),
        mpi::getType<localIdx>(),
        recvIdx.data(),
        sendCounts.data(),
        sdispl.data(),
        mpi::getType<localIdx>(),
        mpiEnviron.comm()
    );

    // FIXME seems unused
    std::vector<localIdx> boundaryMapVector;
    return CommunicationPattern(sendCounts, recvIdx, boundaryMapVector, mpiEnviron);
}

} // namespace NeoN
