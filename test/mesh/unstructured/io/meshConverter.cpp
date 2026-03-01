// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"
#include "testHelpers.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <vtkCellCenters.h>
#include <vtkDataAssembly.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>

using NeoN::test::makeCellConn;


TEST_CASE("MeshConverter face topology")
{
    NeoN::SerialExecutor serial;

    SECTION("Single tet from raw connectivity")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}}, {10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 4);
        REQUIRE(topo.faceOwner.size() == 4);
    }

    SECTION("Two tets sharing a face")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}, {0, 1, 2, 4}}, {10, 10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        REQUIRE(topo.nInternalFaces == 1);
        REQUIRE(topo.nBoundaryFaces == 6); // 4+4-2 = 6
    }

    SECTION("Single hex")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4, 5, 6, 7}}, {12});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 6);
    }

    SECTION("Single pyramid")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4}}, {14});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        REQUIRE(topo.nInternalFaces == 0);
        REQUIRE(topo.nBoundaryFaces == 5); // 1 quad + 4 tri
    }
}


TEST_CASE("MeshConverter geometry")
{
    NeoN::SerialExecutor serial;

    SECTION("Single tet volume and centres")
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}}, {10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        NeoN::Vector<NeoN::Vec3> points(serial, pts);
        auto faceNodesCopy = topo.faceNodes;
        auto geom = NeoN::io::computeGeometry(
            serial,
            points,
            topo.faceOwner,
            topo.faceNeighbour,
            faceNodesCopy,
            topo.nInternalFaces,
            1
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto hVol = hostVol.view();
        auto hCC = hostCC.view();
        REQUIRE(hVol[0] == Catch::Approx(1.0 / 6.0).margin(1e-12));
        REQUIRE(hCC[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(hCC[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(hCC[0][2] == Catch::Approx(0.25).margin(1e-10));
    }

    SECTION("Unit cube hex volume")
    {
        std::vector<NeoN::Vec3> pts = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
        };
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4, 5, 6, 7}}, {12});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        NeoN::Vector<NeoN::Vec3> points(serial, pts);
        auto faceNodesCopy = topo.faceNodes;
        auto geom = NeoN::io::computeGeometry(
            serial,
            points,
            topo.faceOwner,
            topo.faceNeighbour,
            faceNodesCopy,
            topo.nInternalFaces,
            1
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto hVol = hostVol.view();
        auto hCC = hostCC.view();
        REQUIRE(hVol[0] == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(hCC[0][0] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(hCC[0][1] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(hCC[0][2] == Catch::Approx(0.5).margin(1e-10));
    }

    SECTION("Two tets total volume")
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}};
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}, {0, 1, 2, 4}}, {10, 10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        NeoN::Vector<NeoN::Vec3> points(serial, pts);
        auto faceNodesCopy = topo.faceNodes;
        auto geom = NeoN::io::computeGeometry(
            serial,
            points,
            topo.faceOwner,
            topo.faceNeighbour,
            faceNodesCopy,
            topo.nInternalFaces,
            2
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hVol = hostVol.view();
        NeoN::scalar totalVol = hVol[0] + hVol[1];
        REQUIRE(totalVol == Catch::Approx(2.0 / 6.0).margin(1e-12));
    }
}


TEST_CASE("MeshConverter rebuild connectivity")
{
    NeoN::SerialExecutor serial;

    SECTION("Round-trip: build topology then rebuild connectivity")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}}, {10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);

        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto rebuilt = NeoN::io::rebuildCellConnectivity(
            serial,
            topo.faceOwner,
            topo.faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            nFaces
        );

        REQUIRE(rebuilt.nCells == 1);
        auto hostTypes = rebuilt.cellTypes.copyToHost();
        REQUIRE(hostTypes.view()[0] == 10); // VTK_TETRA
        // Should have 4 unique nodes
        auto hostCTN = rebuilt.cellToNodes.copyToHost();
        auto [s0, e0] = hostCTN.view().bounds(0);
        REQUIRE(e0 - s0 == 4);
    }

    SECTION("Hex rebuild preserves type")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4, 5, 6, 7}}, {12});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);

        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto rebuilt = NeoN::io::rebuildCellConnectivity(
            serial,
            topo.faceOwner,
            topo.faceNeighbour,
            topo.faceNodes,
            1,
            topo.nInternalFaces,
            nFaces
        );

        auto hostTypes = rebuilt.cellTypes.copyToHost();
        REQUIRE(hostTypes.view()[0] == 12); // VTK_HEXAHEDRON
        auto hostCTN = rebuilt.cellToNodes.copyToHost();
        auto [s0, e0] = hostCTN.view().bounds(0);
        REQUIRE(e0 - s0 == 8);
    }
}


TEST_CASE("MeshConverter CellInfo rebuild")
{
    NeoN::SerialExecutor serial;

    SECTION("Single tet: CellInfo has cellFaceNodes and correct type")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3}}, {10});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);

        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto cells = NeoN::io::rebuildCellInfo(
            topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 10); // VTK_TETRA
        REQUIRE(cells[0].nodeIds.size() == 4);
        REQUIRE(cells[0].cellFaceNodes.size() == 4); // 4 faces for a tet
    }

    SECTION("Single hex: CellInfo has 6 faces and 8 nodes")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4, 5, 6, 7}}, {12});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);

        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto cells = NeoN::io::rebuildCellInfo(
            topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 12); // VTK_HEXAHEDRON
        REQUIRE(cells[0].nodeIds.size() == 8);
        REQUIRE(cells[0].cellFaceNodes.size() == 6);
    }

    SECTION("Pyramid: CellInfo has 5 faces and 5 nodes")
    {
        auto conn = makeCellConn(serial, {{0, 1, 2, 3, 4}}, {14});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);

        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto cells = NeoN::io::rebuildCellInfo(
            topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
        );

        REQUIRE(cells.size() == 1);
        REQUIRE(cells[0].cellType == 14); // VTK_PYRAMID
        REQUIRE(cells[0].nodeIds.size() == 5);
        REQUIRE(cells[0].cellFaceNodes.size() == 5);
    }
}


TEST_CASE("MeshConverter node ordering")
{
    NeoN::SerialExecutor serial;

    auto makeCellInfo = [&](const std::vector<NeoN::localIdx>& cellNodes, int vtkType)
    {
        auto conn = makeCellConn(serial, {cellNodes}, {static_cast<int32_t>(vtkType)});
        auto topo = NeoN::io::buildFaceTopology(serial, conn);
        auto nFaces = static_cast<NeoN::localIdx>(topo.faceOwner.size());
        auto cells = NeoN::io::rebuildCellInfo(
            topo.faceOwner, topo.faceNeighbour, topo.faceNodes, 1, topo.nInternalFaces, nFaces
        );
        return cells[0];
    };

    SECTION("orderTetNodes returns 4 nodes, all unique, 0-based")
    {
        auto cell = makeCellInfo({0, 1, 2, 3}, 10);
        auto ordered = NeoN::io::orderTetNodes(cell);
        REQUIRE(ordered.size() == 4);

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

        std::set<NeoN::localIdx> baseNodes(ordered.begin(), ordered.begin() + 4);
        REQUIRE(baseNodes.size() == 4);
    }

    SECTION("orderWedgeNodes returns 6 nodes, all unique, 0-based")
    {
        auto cell = makeCellInfo({0, 1, 2, 3, 4, 5}, 13);
        auto ordered = NeoN::io::orderWedgeNodes(cell);
        REQUIRE(ordered.size() == 6);

        std::set<NeoN::localIdx> nodeSet(ordered.begin(), ordered.end());
        REQUIRE(nodeSet.size() == 6);
        REQUIRE(nodeSet == std::set<NeoN::localIdx>({0, 1, 2, 3, 4, 5}));
    }
}
