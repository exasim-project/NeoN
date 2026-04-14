// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


TEST_CASE("Unstructured Mesh")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Can create single cell mesh " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);

        REQUIRE(mesh.nCells() == 1);
        REQUIRE(mesh.nBoundaryFaces() == 4);
        REQUIRE(mesh.nInternalFaces() == 0);
        REQUIRE(mesh.nBoundaries() == 4);
    }

    SECTION("Can create a 1D uniform mesh" + execName)
    {
        NeoN::localIdx nCells = 4;

        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, nCells);

        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 2);
        REQUIRE(mesh.nInternalFaces() == 3);
        REQUIRE(mesh.nBoundaries() == 2);
        REQUIRE(mesh.nTotalFaces() == 5);

        // Verify mesh points
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0
        auto pointsExp = std::vector<NeoN::Vec3> {
            {0.25, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}
        };
        REQUIRE_THAT(pointsExp, IsEqualTo(mesh.points(), EqualInt()));

        // Verify cell centers
        auto cellCentresExp = std::vector<NeoN::Vec3> {
            {0.125, 0.0, 0.0}, {0.375, 0.0, 0.0}, {0.625, 0.0, 0.0}, {0.875, 0.0, 0.0}
        };
        REQUIRE_THAT(cellCentresExp, IsEqualTo(mesh.cellCentres(), EqualInt()));

        // Verify face owners
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto faceOwnerExp = std::vector<NeoN::label> {0, 1, 2, 0, 3};
        REQUIRE_THAT(faceOwnerExp, IsEqualTo(mesh.faceOwner(), EqualInt()));

        // Verify face neighbors
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto faceNeighbourExp = std::vector<NeoN::label> {1, 2, 3};
        REQUIRE_THAT(faceNeighbourExp, IsEqualTo(mesh.faceNeighbour(), EqualInt()));

        // Verify boundary mesh
        auto faceCellsExp = std::vector<NeoN::localIdx> {0, 3};
        REQUIRE_THAT(faceCellsExp, IsEqualTo(mesh.boundaryMesh().faceCells(), EqualInt()));

        auto cnExp = std::vector<NeoN::Vec3> {{0.125, 0.0, 0.0}, {0.875, 0.0, 0.0}};
        REQUIRE_THAT(cnExp, IsEqualTo(mesh.boundaryMesh().cn(), EqualInt()));

        auto deltaExp = std::vector<NeoN::Vec3> {{-0.125, 0.0, 0.0}, {0.125, 0.0, 0.0}};
        REQUIRE_THAT(deltaExp, IsEqualTo(mesh.boundaryMesh().delta(), EqualInt()));
    }
}
