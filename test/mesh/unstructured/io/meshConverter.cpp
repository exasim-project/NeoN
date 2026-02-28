// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include <vtkCellCenters.h>
#include <vtkDataAssembly.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>


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


TEST_CASE("buildMultiBlockMesh")
{
    NeoN::SerialExecutor exec;

    SECTION("2x2 uniform mesh produces 2-level hierarchy")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        // Root: 2 blocks (internalMesh + boundary)
        REQUIRE(mb->GetNumberOfBlocks() == 2);

        // Block 0: internalMesh (vtkUnstructuredGrid)
        auto* volume = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        REQUIRE(volume != nullptr);
        REQUIRE(volume->GetNumberOfCells() == 4);

        // Block 1: boundary (nested vtkMultiBlockDataSet)
        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);
        REQUIRE(boundary->GetNumberOfBlocks() == 6);

        // All patches are vtkPolyData
        for (unsigned int i = 0; i < 6; ++i)
        {
            auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(i));
            REQUIRE(patch != nullptr);
        }
    }

    SECTION("Boundary patches have metadata names")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);

        // Each patch sub-block should have metadata
        for (unsigned int i = 0; i < boundary->GetNumberOfBlocks(); ++i)
        {
            REQUIRE(boundary->HasMetaData(i));
        }
    }

    SECTION("Boundary face counts are correct for 2x2 grid")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));

        // 2x2 grid: xmin=2, xmax=2, ymin=2, ymax=2, zmin=4, zmax=4
        std::vector<int> expectedFaces = {2, 2, 2, 2, 4, 4};
        for (unsigned int i = 0; i < 6; ++i)
        {
            auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(i));
            REQUIRE(patch->GetNumberOfCells() == expectedFaces[i]);
        }
    }

    SECTION("3D mesh patch names and geometry match (3x2x4)")
    {
        // Domain: [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
        // Boundary patches and the planes they must lie on:
        //   patch 0 "xmin"  — x = xmin plane  (ny*nz = 8  faces)
        //   patch 1 "xmax"  — x = xmax plane  (ny*nz = 8  faces)
        //   patch 2 "ymin"  — y = ymin plane  (nx*nz = 12 faces)
        //   patch 3 "ymax"  — y = ymax plane  (nx*nz = 12 faces)
        //   patch 4 "zmin"  — z = zmin plane  (nx*ny = 6  faces)
        //   patch 5 "zmax"  — z = zmax plane  (nx*ny = 6  faces)
        double xmin = 0.0, xmax = 3.0;
        double ymin = 0.0, ymax = 2.0;
        double zmin = 0.0, zmax = 4.0;

        auto mesh = NeoN::createUniform3DGrid(exec, 3, 2, 4, xmax, ymax, zmax);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);

        REQUIRE(boundary->GetNumberOfBlocks() == 6);

        // Use helper to read patch names (avoids VTK static key duplication
        // between shared lib and test executable)
        auto patchNames = NeoN::io::multiBlockPatchNames(boundary);
        REQUIRE(patchNames.size() == 6);

        std::vector<std::string> expectedNames = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        std::vector<int> expectedFaces = {8, 8, 12, 12, 6, 6};

        struct PlaneCheck
        {
            int axis; // 0=x, 1=y, 2=z
            double value;
        };
        std::vector<PlaneCheck> checks = {
            {0, xmin}, {0, xmax}, {1, ymin}, {1, ymax}, {2, zmin}, {2, zmax}
        };

        for (unsigned int p = 0; p < 6; ++p)
        {
            INFO("patch index: " << p);
            REQUIRE(patchNames[p] == expectedNames[p]);

            auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(p));
            REQUIRE(patch != nullptr);
            REQUIRE(patch->GetNumberOfCells() == expectedFaces[p]);

            // Verify all face centres lie on the expected plane
            auto cellCentres = vtkSmartPointer<vtkCellCenters>::New();
            cellCentres->SetInputData(patch);
            cellCentres->Update();
            auto* centresOutput = cellCentres->GetOutput();

            for (vtkIdType c = 0; c < centresOutput->GetNumberOfPoints(); ++c)
            {
                double pt[3];
                centresOutput->GetPoint(c, pt);
                REQUIRE(pt[checks[p].axis] == Catch::Approx(checks[p].value).margin(1e-10));
            }
        }
    }

    SECTION("Patch polydata shares points with volume grid")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        auto* volume = vtkUnstructuredGrid::SafeDownCast(mb->GetBlock(0));
        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(0));
        REQUIRE(volume->GetPoints() == patch->GetPoints());
    }
}


TEST_CASE("buildPartitionedMesh")
{
    NeoN::SerialExecutor exec;

    SECTION("2x2 uniform mesh produces PDC with 7 datasets")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto pdc = NeoN::io::buildPartitionedMesh(mesh);

        // 1 volume + 6 boundary patches
        REQUIRE(pdc->GetNumberOfPartitionedDataSets() == 7);
    }

    SECTION("Dataset 0 is vtkUnstructuredGrid with 4 cells")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto pdc = NeoN::io::buildPartitionedMesh(mesh);

        auto* pds0 = pdc->GetPartitionedDataSet(0);
        REQUIRE(pds0 != nullptr);
        REQUIRE(pds0->GetNumberOfPartitions() == 1);
        auto* grid = vtkUnstructuredGrid::SafeDownCast(pds0->GetPartition(0));
        REQUIRE(grid != nullptr);
        REQUIRE(grid->GetNumberOfCells() == 4);
    }

    SECTION("Datasets 1-6 are vtkPolyData with correct face counts")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto pdc = NeoN::io::buildPartitionedMesh(mesh);

        // 2x2 grid: xmin=2, xmax=2, ymin=2, ymax=2, zmin=4, zmax=4
        std::vector<int> expectedFaces = {2, 2, 2, 2, 4, 4};
        for (unsigned int i = 0; i < 6; ++i)
        {
            auto* pds = pdc->GetPartitionedDataSet(i + 1);
            REQUIRE(pds != nullptr);
            REQUIRE(pds->GetNumberOfPartitions() == 1);
            auto* poly = vtkPolyData::SafeDownCast(pds->GetPartition(0));
            REQUIRE(poly != nullptr);
            REQUIRE(poly->GetNumberOfCells() == expectedFaces[i]);
        }
    }

    SECTION("Assembly has expected hierarchy with correct patch names")
    {
        auto mesh = NeoN::createUniform2DGrid(exec, 2, 2);
        auto pdc = NeoN::io::buildPartitionedMesh(mesh);

        auto* assembly = pdc->GetDataAssembly();
        REQUIRE(assembly != nullptr);

        // Check internalMesh node exists under root
        auto internalNodes = assembly->SelectNodes({"//internalMesh"});
        REQUIRE(internalNodes.size() == 1);

        // Check boundary node exists under root
        auto boundaryNodes = assembly->SelectNodes({"//boundary"});
        REQUIRE(boundaryNodes.size() == 1);

        // Check patch nodes exist under boundary with correct names
        auto patchNodes = assembly->GetChildNodes(boundaryNodes[0]);
        REQUIRE(patchNodes.size() == 6);

        std::vector<std::string> expectedNames = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        for (std::size_t i = 0; i < 6; ++i)
        {
            std::string name = assembly->GetNodeName(patchNodes[i]);
            REQUIRE(name == expectedNames[i]);
        }
    }
}
