// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

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

    // FIXME this should be a test Field
    // SECTION("Can create domainVector from mesh " + execName)
    // {

    //     NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
    //     NeoN::Field<NeoN::scalar> domainVector(exec, mesh.);

    //     REQUIRE(domainVector.boundaryVector().offset().size() == 5);
    // }

    SECTION("Can create a 1D uniform mesh" + execName)
    {
        size_t nCells = 4;

        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, nCells);

        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 2);
        REQUIRE(mesh.nInternalFaces() == 3);
        REQUIRE(mesh.nBoundaries() == 2);
        REQUIRE(mesh.nFaces() == 5);

        // Verify mesh points
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.view()[0][0] == 0.25);
        REQUIRE(hostPoints.view()[1][0] == 0.5);
        REQUIRE(hostPoints.view()[3][0] == 0.0);
        REQUIRE(hostPoints.view()[nCells][0] == 1.0);

        // Verify cell centers
        auto hostCellCentres = mesh.cellCentres().copyToHost();
        REQUIRE(hostCellCentres.view()[0][0] == 0.125);
        REQUIRE(hostCellCentres.view()[1][0] == 0.375);
        REQUIRE(hostCellCentres.view()[2][0] == 0.625);
        REQUIRE(hostCellCentres.view()[3][0] == 0.875);

        // Verify face owners
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceOwner = mesh.faceOwner().copyToHost();
        REQUIRE(hostFaceOwner.view()[0] == 0);
        REQUIRE(hostFaceOwner.view()[1] == 1);
        REQUIRE(hostFaceOwner.view()[2] == 2);
        REQUIRE(hostFaceOwner.view()[3] == 0);
        REQUIRE(hostFaceOwner.view()[4] == 3);

        // Verify face neighbors
        // |_3 0 |_0 1 |_1 2 |_2 3 |_4
        auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
        REQUIRE(hostFaceNeighbour.view()[0] == 1);
        REQUIRE(hostFaceNeighbour.view()[1] == 2);
        REQUIRE(hostFaceNeighbour.view()[2] == 3);

        // Verify boundary mesh
        auto hostBoundaryFaceCells = mesh.boundaryMesh().faceCells().copyToHost();
        auto hostBoundaryCn = mesh.boundaryMesh().cn().copyToHost();
        auto hostBoundaryDelta = mesh.boundaryMesh().delta().copyToHost();
        REQUIRE(hostBoundaryFaceCells.view()[0] == 0);
        REQUIRE(hostBoundaryFaceCells.view()[1] == 3);
        REQUIRE(hostBoundaryCn.view()[0][0] == 0.125);
        REQUIRE(hostBoundaryCn.view()[1][0] == 0.875);
        REQUIRE(hostBoundaryDelta.view()[0][0] == -0.125);
        REQUIRE(hostBoundaryDelta.view()[1][0] == 0.125);
    }
}
