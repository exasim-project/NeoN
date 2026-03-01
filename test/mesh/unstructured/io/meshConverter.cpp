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
        auto geom = NeoN::io::computeGeometry(pts, topo, 1);

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
        auto geom = NeoN::io::computeGeometry(pts, topo, 1);

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
        auto geom = NeoN::io::computeGeometry(pts, topo, 2);

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

        std::vector<int> expectedFaces = {2, 2, 2, 2, 4, 4};
        for (unsigned int i = 0; i < 6; ++i)
        {
            auto* patch = vtkPolyData::SafeDownCast(boundary->GetBlock(i));
            REQUIRE(patch->GetNumberOfCells() == expectedFaces[i]);
        }
    }

    SECTION("3D mesh patch names and geometry match (3x2x4)")
    {
        double xmin = 0.0, xmax = 3.0;
        double ymin = 0.0, ymax = 2.0;
        double zmin = 0.0, zmax = 4.0;

        auto mesh = NeoN::createUniform3DGrid(exec, 3, 2, 4, xmax, ymax, zmax);
        auto mb = NeoN::io::buildMultiBlockMesh(mesh);

        auto* boundary = vtkMultiBlockDataSet::SafeDownCast(mb->GetBlock(1));
        REQUIRE(boundary != nullptr);

        REQUIRE(boundary->GetNumberOfBlocks() == 6);

        auto patchNames = NeoN::io::multiBlockPatchNames(boundary);
        REQUIRE(patchNames.size() == 6);

        std::vector<std::string> expectedNames = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        std::vector<int> expectedFaces = {8, 8, 12, 12, 6, 6};

        struct PlaneCheck
        {
            int axis;
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

        auto internalNodes = assembly->SelectNodes({"//internalMesh"});
        REQUIRE(internalNodes.size() == 1);

        auto boundaryNodes = assembly->SelectNodes({"//boundary"});
        REQUIRE(boundaryNodes.size() == 1);

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
