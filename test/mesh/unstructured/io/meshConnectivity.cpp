// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"

#include <set>


namespace
{

// Helper: build a single-tet CellConnectivity on the given executor.
// Nodes 0,1,2,3 — VTK type 10 (VTK_TETRA).
NeoN::io::CellConnectivity makeTetConn(const NeoN::Executor& exec)
{
    std::vector<NeoN::localIdx> values = {0, 1, 2, 3};
    std::vector<NeoN::localIdx> offsets = {0, 4}; // one segment [0,4)
    NeoN::Vector<NeoN::localIdx> valVec(exec, values);
    NeoN::Vector<NeoN::localIdx> offVec(exec, offsets);

    return NeoN::io::CellConnectivity {
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx>(valVec, offVec),
        NeoN::Vector<int32_t>(exec, std::vector<int32_t> {10}),
        1
    };
}

} // anonymous namespace


TEST_CASE("CellConnectivity uses NeoN SegmentedVector and Vector types")
{
    NeoN::SerialExecutor exec;
    auto conn = makeTetConn(exec);

    REQUIRE(conn.nCells == 1);
    REQUIRE(conn.cellTypes.size() == 1);
    REQUIRE(conn.cellToNodes.numSegments() == 1);
}


TEST_CASE("buildFaceTopology takes Executor and returns NeoN FaceTopology")
{
    NeoN::SerialExecutor exec;
    auto conn = makeTetConn(exec);
    auto topo = NeoN::io::buildFaceTopology(exec, conn);

    REQUIRE(topo.nInternalFaces == 0);
    REQUIRE(topo.nBoundaryFaces == 4);
    REQUIRE(topo.faceOwner.size() == 4);
    REQUIRE(topo.faceNeighbour.size() == 0);
    REQUIRE(topo.faceNodes.numSegments() == 4);
}


TEST_CASE("rebuildCellConnectivity takes NeoN types as input and output")
{
    NeoN::SerialExecutor exec;
    auto conn = makeTetConn(exec);
    auto topo = NeoN::io::buildFaceTopology(exec, conn);

    NeoN::localIdx nFaces = topo.nInternalFaces + topo.nBoundaryFaces;
    auto rebuilt = NeoN::io::rebuildCellConnectivity(
        exec, topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
    );

    REQUIRE(rebuilt.nCells == 1);
    auto hostTypes = rebuilt.cellTypes.copyToHost();
    REQUIRE(hostTypes.view()[0] == 10);
}


TEST_CASE("rebuildCellInfo takes NeoN types")
{
    NeoN::SerialExecutor exec;
    auto conn = makeTetConn(exec);
    auto topo = NeoN::io::buildFaceTopology(exec, conn);

    NeoN::localIdx nFaces = topo.nInternalFaces + topo.nBoundaryFaces;
    auto cells = NeoN::io::rebuildCellInfo(
        topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
    );

    REQUIRE(cells.size() == 1);
    REQUIRE(cells[0].cellType == 10);
    REQUIRE(cells[0].nodeIds.size() == 4);
    REQUIRE(cells[0].cellFaceNodes.size() == 4);
}


TEST_CASE("node ordering functions work with CellInfo from NeoN-typed topology")
{
    NeoN::SerialExecutor exec;
    auto conn = makeTetConn(exec);
    auto topo = NeoN::io::buildFaceTopology(exec, conn);

    NeoN::localIdx nFaces = topo.nInternalFaces + topo.nBoundaryFaces;
    auto cells = NeoN::io::rebuildCellInfo(
        topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
    );

    auto ordered = NeoN::io::orderTetNodes(cells[0]);
    REQUIRE(ordered.size() == 4);

    std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
    REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3}));
}
