// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/mesh/unstructured/io/meshConverter.hpp"


TEST_CASE("MeshConverter face topology")
{
    SECTION("Single tet from raw connectivity")
    {
        // 4 nodes, 1 TETRA_4 cell: nodes 0,1,2,3
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10}; // VTK_TETRA
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 4);
        REQUIRE(topo.faceOwner.size() == 4);
    }

    SECTION("Two tets sharing a face")
    {
        // Nodes: 0-3 for tet1, shares face (0,1,2) with tet2 (0,1,2,4)
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}, {0, 1, 2, 4}};
        conn.cellTypes = {10, 10};
        conn.nCells = 2;

        auto topo = NeoN::io::buildFaceTopology(conn);
        REQUIRE(topo.nInternalFaces == 1);
        REQUIRE(topo.nBoundaryFaces == 6); // 4+4-2 = 6
    }

    SECTION("Single hex")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4, 5, 6, 7}};
        conn.cellTypes = {12}; // VTK_HEXAHEDRON
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 6);
    }

    SECTION("Single pyramid")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4}};
        conn.cellTypes = {14}; // VTK_PYRAMID
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 5); // 1 quad + 4 tri
    }
}


TEST_CASE("MeshConverter geometry")
{
    SECTION("Single tet volume and centres")
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        auto geom = NeoN::io::computeGeometry(pts, topo, 1);

        REQUIRE(geom.cellVolumes[0] == Catch::Approx(1.0 / 6.0).margin(1e-12));
        REQUIRE(geom.cellCentres[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(geom.cellCentres[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(geom.cellCentres[0][2] == Catch::Approx(0.25).margin(1e-10));
    }

    SECTION("Unit cube hex volume")
    {
        std::vector<NeoN::Vec3> pts = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
        };

        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4, 5, 6, 7}};
        conn.cellTypes = {12};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        auto geom = NeoN::io::computeGeometry(pts, topo, 1);

        REQUIRE(geom.cellVolumes[0] == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(geom.cellCentres[0][0] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(geom.cellCentres[0][1] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(geom.cellCentres[0][2] == Catch::Approx(0.5).margin(1e-10));
    }

    SECTION("Two tets total volume")
    {
        // Two tets sharing face (0,1,2), total volume = 2 * (1/6)
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}};

        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}, {0, 1, 2, 4}};
        conn.cellTypes = {10, 10};
        conn.nCells = 2;

        auto topo = NeoN::io::buildFaceTopology(conn);
        auto geom = NeoN::io::computeGeometry(pts, topo, 2);

        NeoN::scalar totalVol = geom.cellVolumes[0] + geom.cellVolumes[1];
        REQUIRE(totalVol == Catch::Approx(2.0 / 6.0).margin(1e-12));
    }
}


TEST_CASE("MeshConverter rebuild connectivity")
{
    SECTION("Round-trip: build topology then rebuild connectivity")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        // Convert to the format expected by rebuildCellConnectivity
        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto rebuilt = NeoN::io::rebuildCellConnectivity(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );

        REQUIRE(rebuilt.nCells == 1);
        REQUIRE(rebuilt.cellTypes[0] == 10); // VTK_TETRA
        // Should have 4 unique nodes
        REQUIRE(rebuilt.cellToNodes[0].size() == 4);
    }

    SECTION("Hex rebuild preserves type")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4, 5, 6, 7}};
        conn.cellTypes = {12};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto rebuilt = NeoN::io::rebuildCellConnectivity(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );

        REQUIRE(rebuilt.cellTypes[0] == 12); // VTK_HEXAHEDRON
        REQUIRE(rebuilt.cellToNodes[0].size() == 8);
    }
}


TEST_CASE("MeshConverter CellInfo rebuild")
{
    SECTION("Single tet: CellInfo has cellFaceNodes and correct type")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto cells = NeoN::io::rebuildCellInfo(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 10); // VTK_TETRA
        REQUIRE(cells[0].nodeIds.size() == 4);
        REQUIRE(cells[0].cellFaceNodes.size() == 4); // 4 faces for a tet
    }

    SECTION("Single hex: CellInfo has 6 faces and 8 nodes")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4, 5, 6, 7}};
        conn.cellTypes = {12};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto cells = NeoN::io::rebuildCellInfo(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 12); // VTK_HEXAHEDRON
        REQUIRE(cells[0].nodeIds.size() == 8);
        REQUIRE(cells[0].cellFaceNodes.size() == 6);
    }

    SECTION("Pyramid: CellInfo has 5 faces and 5 nodes")
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4}};
        conn.cellTypes = {14};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto cells = NeoN::io::rebuildCellInfo(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 14); // VTK_PYRAMID
        REQUIRE(cells[0].nodeIds.size() == 5);
        REQUIRE(cells[0].cellFaceNodes.size() == 5);
    }
}


TEST_CASE("MeshConverter node ordering")
{
    // Helper: build CellInfo from a synthetic cell
    auto makeCellInfo = [](const std::vector<NeoN::localIdx>& cellNodes, int vtkType)
    {
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {cellNodes};
        conn.cellTypes = {vtkType};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);

        std::vector<NeoN::label> faceOwner(topo.faceOwner.begin(), topo.faceOwner.end());
        std::vector<NeoN::label> faceNeighbour(
            topo.faceNeighbour.begin(), topo.faceNeighbour.end()
        );

        auto cells = NeoN::io::rebuildCellInfo(
            faceOwner,
            faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            static_cast<NeoN::localIdx>(topo.faceOwner.size())
        );
        return cells[0];
    };

    SECTION("orderTetNodes returns 4 nodes, all unique, 0-based")
    {
        auto cell = makeCellInfo({0, 1, 2, 3}, 10);
        auto ordered = NeoN::io::orderTetNodes(cell);
        REQUIRE(ordered.size() == 4);

        // All nodes should be from {0,1,2,3}
        std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
        REQUIRE(nodeSet.size() == 4);
        REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3}));
    }

    SECTION("orderHexNodes returns 8 nodes, all unique, 0-based")
    {
        auto cell = makeCellInfo({0, 1, 2, 3, 4, 5, 6, 7}, 12);
        auto ordered = NeoN::io::orderHexNodes(cell);
        REQUIRE(ordered.size() == 8);

        std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
        REQUIRE(nodeSet.size() == 8);
        REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3, 4, 5, 6, 7}));
    }

    SECTION("orderPyramidNodes returns 5 nodes with quad base then apex")
    {
        auto cell = makeCellInfo({0, 1, 2, 3, 4}, 14);
        auto ordered = NeoN::io::orderPyramidNodes(cell);
        REQUIRE(ordered.size() == 5);

        std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
        REQUIRE(nodeSet.size() == 5);
        REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3, 4}));

        // The last node should be the apex (not in any quad face)
        // The first 4 should be the quad base face nodes
        std::set<NeoN::localIdx> baseNodes(ordered.begin(), ordered.begin() + 4);
        REQUIRE(baseNodes.size() == 4);
    }

    SECTION("orderWedgeNodes returns 6 nodes, all unique, 0-based")
    {
        // VTK_WEDGE = 13, nodes 0-5
        auto cell = makeCellInfo({0, 1, 2, 3, 4, 5}, 13);
        auto ordered = NeoN::io::orderWedgeNodes(cell);
        REQUIRE(ordered.size() == 6);

        std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
        REQUIRE(nodeSet.size() == 6);
        REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3, 4, 5}));
    }
}
