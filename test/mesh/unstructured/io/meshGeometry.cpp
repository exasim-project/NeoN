// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"
#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"


TEST_CASE("meshGeometry header provides MeshGeometry and computeGeometry")
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
}
