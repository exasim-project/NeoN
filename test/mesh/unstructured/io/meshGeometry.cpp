// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"
#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"


TEST_CASE("computeFaceCentres")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single triangle face " + execName)
    {
        // Nodes: (0,0,0), (1,0,0), (0,1,0) -> centre = (1/3, 1/3, 0)
        NeoN::Vector<NeoN::Vec3> points(
            exec, std::vector<NeoN::Vec3> {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
        );
        NeoN::Vector<NeoN::localIdx> fnValues(exec, std::vector<NeoN::localIdx> {0, 1, 2});
        NeoN::Vector<NeoN::localIdx> fnSegments(exec, std::vector<NeoN::localIdx> {0, 3});
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(fnValues, fnSegments);

        auto faceCentres = NeoN::io::computeFaceCentres(exec, points, faceNodes);

        auto hostFC = faceCentres.copyToHost();
        auto fcView = hostFC.view();
        REQUIRE(fcView[0][0] == Catch::Approx(1.0 / 3.0).margin(1e-12));
        REQUIRE(fcView[0][1] == Catch::Approx(1.0 / 3.0).margin(1e-12));
        REQUIRE(fcView[0][2] == Catch::Approx(0.0).margin(1e-12));
    }

    SECTION("Quad face " + execName)
    {
        // Nodes: (0,0,0), (1,0,0), (1,1,0), (0,1,0) -> centre = (0.5, 0.5, 0)
        NeoN::Vector<NeoN::Vec3> points(
            exec, std::vector<NeoN::Vec3> {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}
        );
        NeoN::Vector<NeoN::localIdx> fnValues(exec, std::vector<NeoN::localIdx> {0, 1, 2, 3});
        NeoN::Vector<NeoN::localIdx> fnSegments(exec, std::vector<NeoN::localIdx> {0, 4});
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(fnValues, fnSegments);

        auto faceCentres = NeoN::io::computeFaceCentres(exec, points, faceNodes);

        auto hostFC = faceCentres.copyToHost();
        auto fcView = hostFC.view();
        REQUIRE(fcView[0][0] == Catch::Approx(0.5).margin(1e-12));
        REQUIRE(fcView[0][1] == Catch::Approx(0.5).margin(1e-12));
        REQUIRE(fcView[0][2] == Catch::Approx(0.0).margin(1e-12));
    }
}


TEST_CASE("computeFaceAreas")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Unit square face area " + execName)
    {
        // Square face in XY plane: area = (0, 0, 1)
        NeoN::Vector<NeoN::Vec3> points(
            exec, std::vector<NeoN::Vec3> {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}
        );
        NeoN::Vector<NeoN::localIdx> fnValues(exec, std::vector<NeoN::localIdx> {0, 1, 2, 3});
        NeoN::Vector<NeoN::localIdx> fnSegments(exec, std::vector<NeoN::localIdx> {0, 4});
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(fnValues, fnSegments);

        auto faceCentres = NeoN::io::computeFaceCentres(exec, points, faceNodes);
        auto faceAreas = NeoN::io::computeFaceAreas(exec, points, faceNodes, faceCentres);

        auto hostFA = faceAreas.copyToHost();
        auto faView = hostFA.view();
        REQUIRE(faView[0][0] == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(faView[0][1] == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(faView[0][2] == Catch::Approx(1.0).margin(1e-12));
    }
}


TEST_CASE("computeMagFaceAreas")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Magnitude matches mag() " + execName)
    {
        NeoN::Vector<NeoN::Vec3> faceAreas(
            exec, std::vector<NeoN::Vec3> {{0.0, 0.0, 1.0}, {3.0, 4.0, 0.0}}
        );

        auto magFA = NeoN::io::computeMagFaceAreas(exec, faceAreas);

        auto hostMag = magFA.copyToHost();
        auto magView = hostMag.view();
        REQUIRE(magView[0] == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(magView[1] == Catch::Approx(5.0).margin(1e-12));
    }
}


TEST_CASE("buildCellToFaceMapping")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Two tets sharing a face " + execName)
    {
        // Build topology from connectivity
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}, {0, 1, 2, 4}};
        conn.cellTypes = {10, 10};
        conn.nCells = 2;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        auto cellFaces = NeoN::io::buildCellToFaceMapping(
            exec, faceOwner, faceNeighbour, topo.nInternalFaces, 2
        );

        REQUIRE(cellFaces.numSegments() == 2);

        // Each tet has 4 faces. Cell 0: 4 faces, Cell 1: 4 faces
        auto hostCF = cellFaces.copyToHost();
        auto cfView = hostCF.view();
        auto [s0, e0] = cfView.bounds(0);
        auto [s1, e1] = cfView.bounds(1);
        REQUIRE(e0 - s0 == 4);
        REQUIRE(e1 - s1 == 4);
    }
}


TEST_CASE("computeCellCentres")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single tet cell centre " + execName)
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::Vec3> points(exec, pts);
        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        // Build faceNodes SegmentedVector
        std::vector<NeoN::localIdx> flatValues;
        std::vector<NeoN::localIdx> sizes;
        for (const auto& fn : topo.faceNodes)
        {
            sizes.push_back(static_cast<NeoN::localIdx>(fn.size()));
            flatValues.insert(flatValues.end(), fn.begin(), fn.end());
        }
        NeoN::Vector<NeoN::localIdx> intervals(exec, sizes);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(intervals);
        NeoN::Vector<NeoN::localIdx> flatVals(exec, flatValues);
        auto srcView = flatVals.view();
        auto [dstValues, dstSeg] = faceNodes.views();
        NeoN::parallelFor(
            exec,
            {0, flatVals.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { dstValues[i] = srcView[i]; },
            "copyFN"
        );

        auto faceCentres = NeoN::io::computeFaceCentres(exec, points, faceNodes);
        auto cellFaces = NeoN::io::buildCellToFaceMapping(
            exec, faceOwner, faceNeighbour, topo.nInternalFaces, 1
        );
        auto cellCentres = NeoN::io::computeCellCentres(exec, faceCentres, cellFaces, 1);

        auto hostCC = cellCentres.copyToHost();
        auto ccView = hostCC.view();
        REQUIRE(ccView[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][2] == Catch::Approx(0.25).margin(1e-10));
    }
}


TEST_CASE("computeCellVolumes")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single tet volume = 1/6 " + execName)
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::Vec3> points(exec, pts);
        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        std::vector<NeoN::localIdx> flatValues;
        std::vector<NeoN::localIdx> sizes;
        for (const auto& fn : topo.faceNodes)
        {
            sizes.push_back(static_cast<NeoN::localIdx>(fn.size()));
            flatValues.insert(flatValues.end(), fn.begin(), fn.end());
        }
        NeoN::Vector<NeoN::localIdx> intervals(exec, sizes);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(intervals);
        NeoN::Vector<NeoN::localIdx> flatVals(exec, flatValues);
        auto srcView = flatVals.view();
        auto [dstValues, dstSeg] = faceNodes.views();
        NeoN::parallelFor(
            exec,
            {0, flatVals.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { dstValues[i] = srcView[i]; },
            "copyFN"
        );

        auto faceCentres = NeoN::io::computeFaceCentres(exec, points, faceNodes);
        auto cellFaces = NeoN::io::buildCellToFaceMapping(
            exec, faceOwner, faceNeighbour, topo.nInternalFaces, 1
        );
        auto cellCentres = NeoN::io::computeCellCentres(exec, faceCentres, cellFaces, 1);
        auto cellVolumes = NeoN::io::computeCellVolumes(
            exec, points, faceNodes, faceCentres, cellCentres, cellFaces, 1
        );

        auto hostVol = cellVolumes.copyToHost();
        auto volView = hostVol.view();
        REQUIRE(volView[0] == Catch::Approx(1.0 / 6.0).margin(1e-12));
    }
}


TEST_CASE("computeGeometry full pipeline")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Single tet " + execName)
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::Vec3> points(exec, pts);
        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        std::vector<NeoN::localIdx> flatValues;
        std::vector<NeoN::localIdx> sizes;
        for (const auto& fn : topo.faceNodes)
        {
            sizes.push_back(static_cast<NeoN::localIdx>(fn.size()));
            flatValues.insert(flatValues.end(), fn.begin(), fn.end());
        }
        NeoN::Vector<NeoN::localIdx> intervals(exec, sizes);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(intervals);
        NeoN::Vector<NeoN::localIdx> flatVals(exec, flatValues);
        auto srcView = flatVals.view();
        auto [dstValues, dstSeg] = faceNodes.views();
        NeoN::parallelFor(
            exec,
            {0, flatVals.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { dstValues[i] = srcView[i]; },
            "copyFN"
        );

        auto geom = NeoN::io::computeGeometry(
            exec, points, faceOwner, faceNeighbour, faceNodes, topo.nInternalFaces, 1
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto volView = hostVol.view();
        auto ccView = hostCC.view();
        REQUIRE(volView[0] == Catch::Approx(1.0 / 6.0).margin(1e-12));
        REQUIRE(ccView[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][2] == Catch::Approx(0.25).margin(1e-10));
    }

    SECTION("Unit cube hex " + execName)
    {
        std::vector<NeoN::Vec3> pts = {
            {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
        };
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3, 4, 5, 6, 7}};
        conn.cellTypes = {12};
        conn.nCells = 1;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::Vec3> points(exec, pts);
        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        std::vector<NeoN::localIdx> flatValues;
        std::vector<NeoN::localIdx> sizes;
        for (const auto& fn : topo.faceNodes)
        {
            sizes.push_back(static_cast<NeoN::localIdx>(fn.size()));
            flatValues.insert(flatValues.end(), fn.begin(), fn.end());
        }
        NeoN::Vector<NeoN::localIdx> intervals(exec, sizes);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(intervals);
        NeoN::Vector<NeoN::localIdx> flatVals(exec, flatValues);
        auto srcView = flatVals.view();
        auto [dstValues, dstSeg] = faceNodes.views();
        NeoN::parallelFor(
            exec,
            {0, flatVals.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { dstValues[i] = srcView[i]; },
            "copyFN"
        );

        auto geom = NeoN::io::computeGeometry(
            exec, points, faceOwner, faceNeighbour, faceNodes, topo.nInternalFaces, 1
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto volView = hostVol.view();
        auto ccView = hostCC.view();
        REQUIRE(volView[0] == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(ccView[0][0] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(ccView[0][1] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(ccView[0][2] == Catch::Approx(0.5).margin(1e-10));
    }

    SECTION("Two tets sharing a face " + execName)
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}};
        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}, {0, 1, 2, 4}};
        conn.cellTypes = {10, 10};
        conn.nCells = 2;
        auto topo = NeoN::io::buildFaceTopology(conn);

        NeoN::Vector<NeoN::Vec3> points(exec, pts);
        NeoN::Vector<NeoN::localIdx> faceOwner(exec, topo.faceOwner);
        NeoN::Vector<NeoN::localIdx> faceNeighbour(exec, topo.faceNeighbour);

        std::vector<NeoN::localIdx> flatValues;
        std::vector<NeoN::localIdx> sizes;
        for (const auto& fn : topo.faceNodes)
        {
            sizes.push_back(static_cast<NeoN::localIdx>(fn.size()));
            flatValues.insert(flatValues.end(), fn.begin(), fn.end());
        }
        NeoN::Vector<NeoN::localIdx> intervals(exec, sizes);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> faceNodes(intervals);
        NeoN::Vector<NeoN::localIdx> flatVals(exec, flatValues);
        auto srcView = flatVals.view();
        auto [dstValues, dstSeg] = faceNodes.views();
        NeoN::parallelFor(
            exec,
            {0, flatVals.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { dstValues[i] = srcView[i]; },
            "copyFN"
        );

        auto geom = NeoN::io::computeGeometry(
            exec, points, faceOwner, faceNeighbour, faceNodes, topo.nInternalFaces, 2
        );

        auto hostVol = geom.cellVolumes.copyToHost();
        auto volView = hostVol.view();
        NeoN::scalar totalVol = volView[0] + volView[1];
        REQUIRE(totalVol == Catch::Approx(2.0 / 6.0).margin(1e-12));
    }
}


TEST_CASE("computeGeometry legacy overload")
{
    SECTION("Single tet via legacy API")
    {
        std::vector<NeoN::Vec3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        NeoN::io::CellConnectivity conn;
        conn.cellToNodes = {{0, 1, 2, 3}};
        conn.cellTypes = {10};
        conn.nCells = 1;

        auto topo = NeoN::io::buildFaceTopology(conn);
        auto geom = NeoN::io::computeGeometry(pts, topo, 1);

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto volView = hostVol.view();
        auto ccView = hostCC.view();
        REQUIRE(volView[0] == Catch::Approx(1.0 / 6.0).margin(1e-12));
        REQUIRE(ccView[0][0] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][1] == Catch::Approx(0.25).margin(1e-10));
        REQUIRE(ccView[0][2] == Catch::Approx(0.25).margin(1e-10));
    }

    SECTION("Unit cube hex via legacy API")
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

        auto hostVol = geom.cellVolumes.copyToHost();
        auto hostCC = geom.cellCentres.copyToHost();
        auto volView = hostVol.view();
        auto ccView = hostCC.view();
        REQUIRE(volView[0] == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(ccView[0][0] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(ccView[0][1] == Catch::Approx(0.5).margin(1e-10));
        REQUIRE(ccView[0][2] == Catch::Approx(0.5).margin(1e-10));
    }
}
