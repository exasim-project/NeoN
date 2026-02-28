// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"

#include <set>


TEST_CASE("meshConnectivity header provides CellConnectivity")
{
    NeoN::io::CellConnectivity conn;
    conn.cellToNodes = {{0, 1, 2, 3}};
    conn.cellTypes = {10};
    conn.nCells = 1;

    REQUIRE(conn.nCells == 1);
}


TEST_CASE("meshConnectivity header provides FaceTopology and buildFaceTopology")
{
    NeoN::io::CellConnectivity conn;
    conn.cellToNodes = {{0, 1, 2, 3}};
    conn.cellTypes = {10};
    conn.nCells = 1;

    auto topo = NeoN::io::buildFaceTopology(conn);
    REQUIRE(topo.nInternalFaces == 0);
    REQUIRE(topo.nBoundaryFaces == 4);
}


TEST_CASE("meshConnectivity header provides rebuildCellConnectivity")
{
    NeoN::io::CellConnectivity conn;
    conn.cellToNodes = {{0, 1, 2, 3}};
    conn.cellTypes = {10};
    conn.nCells = 1;

    auto topo = NeoN::io::buildFaceTopology(conn);

    std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
    std::vector<NeoN::label> faceNeighbour(topo.faceNeighbour.begin(), topo.faceNeighbour.end());

    auto rebuilt = NeoN::io::rebuildCellConnectivity(
        faceOwner,
        faceNeighbour,
        topo.faceNodes,
        1,
        topo.nInternalFaces,
        static_cast<NeoN::localIdx>(topo.faceOwner.size())
    );

    REQUIRE(rebuilt.nCells == 1);
    REQUIRE(rebuilt.cellTypes[0] == 10);
}


TEST_CASE("meshConnectivity header provides CellInfo and rebuildCellInfo")
{
    NeoN::io::CellConnectivity conn;
    conn.cellToNodes = {{0, 1, 2, 3}};
    conn.cellTypes = {10};
    conn.nCells = 1;

    auto topo = NeoN::io::buildFaceTopology(conn);

    std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
    std::vector<NeoN::label> faceNeighbour(topo.faceNeighbour.begin(), topo.faceNeighbour.end());

    auto cells = NeoN::io::rebuildCellInfo(
        faceOwner,
        faceNeighbour,
        topo.faceNodes,
        1,
        topo.nInternalFaces,
        static_cast<NeoN::localIdx>(topo.faceOwner.size())
    );

    REQUIRE(cells.size() == 1);
    REQUIRE(cells[0].cellType == 10);
    REQUIRE(cells[0].nodeIds.size() == 4);
    REQUIRE(cells[0].cellFaceNodes.size() == 4);
}


TEST_CASE("meshConnectivity header provides node ordering functions")
{
    NeoN::io::CellConnectivity conn;
    conn.cellToNodes = {{0, 1, 2, 3}};
    conn.cellTypes = {10};
    conn.nCells = 1;

    auto topo = NeoN::io::buildFaceTopology(conn);

    std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
    std::vector<NeoN::label> faceNeighbour(topo.faceNeighbour.begin(), topo.faceNeighbour.end());

    auto cells = NeoN::io::rebuildCellInfo(
        faceOwner,
        faceNeighbour,
        topo.faceNodes,
        1,
        topo.nInternalFaces,
        static_cast<NeoN::localIdx>(topo.faceOwner.size())
    );

    auto ordered = NeoN::io::orderTetNodes(cells[0]);
    REQUIRE(ordered.size() == 4);

    std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
    REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3}));
}
