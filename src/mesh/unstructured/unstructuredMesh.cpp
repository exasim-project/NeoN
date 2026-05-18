// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/mesh/unstructured/uniformMeshDataGenerator.hpp"

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/core/primitives/vec3.hpp" // for Vec3

#include <algorithm>
#include <numeric>


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

localIdx UnstructuredMesh::nProcBoundaryFaces() const { return boundaryMesh_.nProcBoundaryFaces(); }

localIdx UnstructuredMesh::nTotalFaces() const
{
    return nInternalFaces_ + nBoundaryFaces_ + boundaryMesh_.nProcBoundaryFaces();
}

localIdx UnstructuredMesh::globalOffset() const { return globalOffset_; }

void UnstructuredMesh::setGlobalOffset(localIdx offset) { globalOffset_ = offset; }

localIdx UnstructuredMesh::nBoundaries() const { return nBoundaries_; }

localIdx UnstructuredMesh::nFaces() const { return nFaces_; }

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
        1,                    // nCells
        0,                    // nInternalFaces,
        4,                    // nBoundaryFaces,
        4,                    // nBoundaries,
        4,                    // nFaces,
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

/* @brief helper that returns the global cell offset of this rank.
 *
 * Used by distributed meshes (boundaryMesh.isDistributed() == true). For
 * non-distributed meshes returns 0.
 */
namespace detail
{

inline localIdx computeGlobalOffsetForLocal(const BoundaryMesh& boundaryMesh, localIdx localNCells)
{
    if (!boundaryMesh.isDistributed())
    {
        return 0;
    }
    mpi::Environment mpiEnviron;
    const auto nRanks = mpiEnviron.sizeRank();
    const auto myRank = mpiEnviron.rank();
    std::vector<int> allNCells(static_cast<std::size_t>(nRanks));
    int local = static_cast<int>(localNCells);
    MPI_Allgather(&local, 1, MPI_INT, allNCells.data(), 1, MPI_INT, mpiEnviron.comm());
    std::vector<localIdx> globalOffset(static_cast<std::size_t>(nRanks + 1), 0);
    for (int i = 0; i < nRanks; i++)
    {
        globalOffset[static_cast<std::size_t>(i + 1)] =
            globalOffset[static_cast<std::size_t>(i)]
            + static_cast<localIdx>(allNCells[static_cast<std::size_t>(i)]);
    }
    return globalOffset[static_cast<std::size_t>(myRank)];
}

} // namespace detail

/* @brief Build a 1-D mesh slice for the local rank in a multi-rank run.
 *
 * Re-attached from feat/gpu-distributed pre-CooSparsity (commit 1059b7d81c).
 * Builds the local nCells cells, appends regular boundaries first and then
 * one or two proc-boundary patches depending on rank position, sets the
 * mesh's globalOffset and returns the assembled UnstructuredMesh.
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells)
{
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
    if (mpiEnviron.rank() == 0)
    {
        faceCellVec.push_back(0);
        faceCellVec.push_back(nCells - 1);
        boundaryWeights.push_back(-1.0);
        boundaryWeights.push_back(1.0);
        neighRanks.push_back(1);
    }
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
        tmp.boundaryMesh().delta(),       // delta
        {exec, boundaryWeights},          // weights
        tmp.boundaryMesh().deltaCoeffs(), // deltaCoeffs
        {0, 1, 2},                        // offset
        nProcBoundaryFaces,               // number of proc boundary patches
        neighRanks                        // neighbourRank
    );

    // On rank last the last two face centres need to be swapped because proc
    // boundaries come first in the compressed boundary tail layout.
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

    // nBoundaryFaces stored on the mesh excludes proc faces (matches the
    // BoundaryMesh::nBoundaryFaces() semantics). nFaces is the OF-full face
    // count = nInternalFaces + nBoundaryFaces (no proc). Proc-face counts are
    // recovered via mesh.nProcBoundaryFaces() / mesh.nTotalFaces().
    const localIdx nRegularBoundary =
        static_cast<localIdx>(faceCellVec.size()) - nProcBoundaryFaces;
    const localIdx nFacesNonProc = tmp.nInternalFaces() + nRegularBoundary;

    UnstructuredMesh mesh(
        exec,
        tmp.points(),
        tmp.cellVolumes(),
        tmp.cellCentres(),
        faceAreasH.copyToExecutor(exec),
        faceCentresH.copyToExecutor(exec),
        tmp.magFaceAreas(),
        tmp.faceOwner(),
        tmp.faceNeighbour(),
        tmp.nCells(),
        tmp.nInternalFaces(),
        nRegularBoundary,
        static_cast<localIdx>(boundaryMesh.offset().size() - 1), // nBoundaries
        nFacesNonProc,
        boundaryMesh
    );

    mesh.setGlobalOffset(detail::computeGlobalOffsetForLocal(boundaryMesh, tmp.nCells()));

    return mesh;
}

CommunicationPattern computeCommunicationPattern(const UnstructuredMesh& mesh)
{
    mpi::Environment mpiEnviron;
    // early return if not distributed
    if (!mesh.boundaryMesh().isDistributed())
    {
        return {};
    }

    auto sendCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank() + 1), 0);

    const auto neighbourRanks = mesh.boundaryMesh().neighbourRank();
    const auto offsets = mesh.boundaryMesh().offset();
    const auto nInnerBoundaries =
        mesh.boundaryMesh().nBoundaries() - mesh.boundaryMesh().nProcBoundaryPatches();

    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        auto patchSize = static_cast<int>(
            offsets[static_cast<std::size_t>(nInnerBoundaries + i + 1)]
            - offsets[static_cast<std::size_t>(nInnerBoundaries + i)]
        );
        sendCounts[static_cast<std::size_t>(targetRank)] = patchSize;
        sendCounts[static_cast<std::size_t>(mpiEnviron.sizeRank())] += patchSize;
    }

    // Build send buffer in MESH-BOUNDARY order: each proc patch is appended in
    // the order it appears in the boundary mesh (the natural layout of
    // boundaryMesh().faceCells()). recvIdx ends up in MESH-BOUNDARY order on
    // the receiving side, which is what setProcBoundarySparsityPattern expects.
    auto globalOffset = mesh.globalOffset();
    const auto faceCells = mesh.boundaryMesh().faceCells();
    auto faceCellsH = faceCells.copyToHost();

    auto buffer = std::vector<localIdx>();
    buffer.reserve(static_cast<std::size_t>(mesh.boundaryMesh().nProcBoundaryFaces()));
    auto procStart = offsets[static_cast<std::size_t>(nInnerBoundaries)];
    for (int i = 0; i < mesh.boundaryMesh().nProcBoundaryFaces(); i++)
    {
        buffer.push_back(faceCellsH.view()[i + procStart] + globalOffset);
    }

    // For each mesh-order proc patch i with target rank neighbourRanks[i], the
    // displacement into the proc-tail buffer for rank r is the patch start
    // offset relative to procStart. Ranks that don't communicate keep the
    // default 0 displacement.
    auto sdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int i = 0; i < static_cast<int>(neighbourRanks.size()); i++)
    {
        const auto targetRank = static_cast<int>(neighbourRanks[static_cast<std::size_t>(i)]);
        sdispl[static_cast<std::size_t>(targetRank)] =
            static_cast<int>(offsets[static_cast<std::size_t>(nInnerBoundaries + i)] - procStart);
    }
    auto recvIdx = std::vector<int>(buffer.size());

    // derive per-rank recv counts by exchanging send counts.
    auto recvCounts = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, mpiEnviron.comm());
    auto rdispl = std::vector<int>(static_cast<std::size_t>(mpiEnviron.sizeRank()), 0);
    for (int r = 1; r < mpiEnviron.sizeRank(); ++r)
        rdispl[static_cast<std::size_t>(r)] =
            rdispl[static_cast<std::size_t>(r - 1)] + recvCounts[static_cast<std::size_t>(r - 1)];
    int totalRecv = rdispl.back() + recvCounts.back();
    recvIdx.resize(static_cast<std::size_t>(totalRecv));

    // sendBuffer is std::vector<localIdx>; convert to int for MPI_INT exchange.
    std::vector<int> sendBuffer(buffer.size());
    for (std::size_t i = 0; i < buffer.size(); ++i)
        sendBuffer[i] = static_cast<int>(buffer[i]);

    MPI_Alltoallv(
        sendBuffer.data(),
        sendCounts.data(),
        sdispl.data(),
        MPI_INT,
        recvIdx.data(),
        recvCounts.data(),
        rdispl.data(),
        MPI_INT,
        mpiEnviron.comm()
    );

    std::vector<localIdx> boundaryMapVector;
    return CommunicationPattern {sendCounts, recvIdx, boundaryMapVector, mpiEnviron};
}

} // namespace NeoN
