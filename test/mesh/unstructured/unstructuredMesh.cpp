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

    SECTION("Can create a uniform 2D grid (OpenFOAM-style hex slab) " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        auto mesh = NeoN::createUniform2DGrid(exec, nx, ny);

        // Topology: 2x2 grid, one cell thick in z (hex cells)
        // - 4 hex cells
        // - 18 points = (2+1)*(2+1)*2  (two z-planes)
        // - 4 internal faces (quad): 2 vertical + 2 horizontal
        // - 16 boundary faces: left(2) + right(2) + bottom(2) + top(2) + front(4) + back(4)
        // - 6 boundaries: left, right, bottom, top, front, back
        // - 20 total faces
        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nInternalFaces() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 16);
        REQUIRE(mesh.nBoundaries() == 6);
        REQUIRE(mesh.nFaces() == 20);

        // Verify point count: two z-planes
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.size() == 18);

        // Verify cell volumes (each cell is 0.5 * 0.5 * 1.0 = 0.25)
        auto hostVols = mesh.cellVolumes().copyToHost();
        for (NeoN::localIdx c = 0; c < 4; ++c)
            REQUIRE(hostVols.view()[c] == Catch::Approx(0.25));

        // Verify cell centres (z should be 0.5, halfway through the slab)
        auto hostCC = mesh.cellCentres().copyToHost();
        REQUIRE(hostCC.view()[0][0] == Catch::Approx(0.25));
        REQUIRE(hostCC.view()[0][1] == Catch::Approx(0.25));
        REQUIRE(hostCC.view()[0][2] == Catch::Approx(0.5));
        REQUIRE(hostCC.view()[1][0] == Catch::Approx(0.75));
        REQUIRE(hostCC.view()[1][1] == Catch::Approx(0.25));
        REQUIRE(hostCC.view()[3][0] == Catch::Approx(0.75));
        REQUIRE(hostCC.view()[3][1] == Catch::Approx(0.75));

        // Verify boundary face count per patch via boundary mesh offset
        // 6 patches: left(2), right(2), bottom(2), top(2), front(4), back(4)
        auto& bm = mesh.boundaryMesh();
        REQUIRE(bm.faceCells().size() == 16);

        // Verify boundary delta vectors
        // Left boundary: delta.x should be negative (face at x=0, cell centre at x=0.25)
        auto hostBndDelta = bm.delta().copyToHost();
        REQUIRE(hostBndDelta.view()[0][0] == Catch::Approx(-0.25));
        REQUIRE(hostBndDelta.view()[0][1] == Catch::Approx(0.0));
    }

    SECTION("Uniform 2D grid stores face node connectivity in stencilDB " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        auto mesh = NeoN::createUniform2DGrid(exec, nx, ny);

        // stencilDB must contain "io::faceNodes"
        REQUIRE(mesh.stencilDB().contains("io::faceNodes"));

        auto& faceNodes =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<NeoN::localIdx>>>>(
                "io::faceNodes"
            );

        // Must have one entry per face
        REQUIRE(static_cast<NeoN::localIdx>(faceNodes.size()) == mesh.nFaces());

        // Each face is a quad → 4 nodes per face
        for (auto& fn : faceNodes)
            REQUIRE(fn.size() == 4);

        // Verify specific face nodes for first vertical internal face (between cell 0 and cell 1)
        // In a 2x2 grid on [0,1]x[0,1]x[0,1], points are indexed as:
        // Bottom z-plane (k=0): pt(i,j,0) = j*(nx+1) + i
        //   6--7--8      (row j=2)
        //   |  |  |
        //   3--4--5      (row j=1)
        //   |  |  |
        //   0--1--2      (row j=0)
        // Top z-plane (k=1): pt(i,j,1) = (nx+1)*(ny+1) + j*(nx+1) + i  (offset by 9)
        //   15-16-17
        //   |  |  |
        //   12-13-14
        //   |  |  |
        //   9--10-11
        //
        // First vertical internal face (i=0, j=0): between cell(0,0) and cell(1,0)
        // at x=dx, connects bottom pt(1,0,0)=1, pt(1,1,0)=4,
        //                     top pt(1,0,1)=10, pt(1,1,1)=13
        auto& firstVertFace = faceNodes[0];
        std::vector<NeoN::localIdx> sorted(firstVertFace.begin(), firstVertFace.end());
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(sorted[0] == 1);
        REQUIRE(sorted[1] == 4);
        REQUIRE(sorted[2] == 10);
        REQUIRE(sorted[3] == 13);
    }

    SECTION("Uniform 2D grid stores patch names in stencilDB " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        auto mesh = NeoN::createUniform2DGrid(exec, nx, ny);

        // stencilDB must contain "io::patchNames"
        REQUIRE(mesh.stencilDB().contains("io::patchNames"));

        auto& patchNames =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");

        // Must have one entry per boundary (6 patches)
        REQUIRE(patchNames.size() == 6);

        // Patch names must match the boundary order:
        // xmin, xmax, ymin, ymax, zmin, zmax
        REQUIRE(patchNames[0] == "xmin");
        REQUIRE(patchNames[1] == "xmax");
        REQUIRE(patchNames[2] == "ymin");
        REQUIRE(patchNames[3] == "ymax");
        REQUIRE(patchNames[4] == "zmin");
        REQUIRE(patchNames[5] == "zmax");
    }

    SECTION("Can create a uniform 2D grid with non-unit domain " + execName)
    {
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::scalar Lx = 3.0;
        NeoN::scalar Ly = 2.0;
        auto mesh = NeoN::createUniform2DGrid(exec, nx, ny, Lx, Ly);

        // 3x2 grid: 6 hex cells, (3-1)*2 + 3*(2-1) = 4+3 = 7 internal faces
        // boundary: left(2) + right(2) + bottom(3) + top(3) + front(6) + back(6) = 22
        REQUIRE(mesh.nCells() == 6);
        REQUIRE(mesh.nInternalFaces() == 7);
        REQUIRE(mesh.nBoundaryFaces() == 22);
        REQUIRE(mesh.nBoundaries() == 6);
        REQUIRE(mesh.nFaces() == 29);

        // Cell volume = (3/3) * (2/2) * 1.0 = 1.0
        auto hostVols = mesh.cellVolumes().copyToHost();
        for (NeoN::localIdx c = 0; c < 6; ++c)
            REQUIRE(hostVols.view()[c] == Catch::Approx(1.0));
    }

    SECTION("2D grid patch face centres lie on correct planes (3x2) " + execName)
    {
        // Domain: [xmin, xmax] x [ymin, ymax] x [zmin, zmax] (one cell thick in z)
        // Boundary patches and the planes they must lie on:
        //   patch 0 "xmin"  — x = xmin plane  (ny    = 2 faces)
        //   patch 1 "xmax"  — x = xmax plane  (ny    = 2 faces)
        //   patch 2 "ymin"  — y = ymin plane  (nx    = 3 faces)
        //   patch 3 "ymax"  — y = ymax plane  (nx    = 3 faces)
        //   patch 4 "zmin"  — z = zmin plane  (nx*ny = 6 faces)
        //   patch 5 "zmax"  — z = zmax plane  (nx*ny = 6 faces)
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::scalar xmin = 0.0;
        NeoN::scalar xmax = 3.0;
        NeoN::scalar ymin = 0.0;
        NeoN::scalar ymax = 2.0;
        NeoN::scalar zmin = 0.0;
        NeoN::scalar zmax = 1.0; // fixed slab thickness
        auto mesh = NeoN::createUniform2DGrid(exec, nx, ny, xmax, ymax);

        auto& bm = mesh.boundaryMesh();
        auto& offset = bm.offset();
        REQUIRE(offset.size() == 7);
        REQUIRE(offset[1] - offset[0] == 2); // left
        REQUIRE(offset[2] - offset[1] == 2); // right
        REQUIRE(offset[3] - offset[2] == 3); // bottom
        REQUIRE(offset[4] - offset[3] == 3); // top
        REQUIRE(offset[5] - offset[4] == 6); // front
        REQUIRE(offset[6] - offset[5] == 6); // back

        auto hostCf = bm.cf().copyToHost();

        // left (patch 0): all face centres on x = xmin plane
        for (NeoN::localIdx f = offset[0]; f < offset[1]; ++f)
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmin).margin(1e-10));

        // right (patch 1): all face centres on x = xmax plane
        for (NeoN::localIdx f = offset[1]; f < offset[2]; ++f)
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmax).margin(1e-10));

        // bottom (patch 2): all face centres on y = ymin plane
        for (NeoN::localIdx f = offset[2]; f < offset[3]; ++f)
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymin).margin(1e-10));

        // top (patch 3): all face centres on y = ymax plane
        for (NeoN::localIdx f = offset[3]; f < offset[4]; ++f)
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymax).margin(1e-10));

        // front (patch 4): all face centres on z = zmin plane
        for (NeoN::localIdx f = offset[4]; f < offset[5]; ++f)
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmin).margin(1e-10));

        // back (patch 5): all face centres on z = zmax plane
        for (NeoN::localIdx f = offset[5]; f < offset[6]; ++f)
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmax).margin(1e-10));
    }

    SECTION("Can create a uniform 3D grid (2x2x2) " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 2;
        auto mesh = NeoN::createUniform3DGrid(exec, nx, ny, nz);

        // Topology: 2x2x2 grid
        // - 8 hex cells
        // - internal faces: x:(2-1)*2*2=4, y:2*(2-1)*2=4, z:2*2*(2-1)=4 → 12 total
        // - boundary faces: left(4)+right(4)+bottom(4)+top(4)+front(4)+back(4) = 24
        // - 6 boundaries
        // - 36 total faces
        // - 27 points = (2+1)*(2+1)*(2+1)
        REQUIRE(mesh.nCells() == 8);
        REQUIRE(mesh.nInternalFaces() == 12);
        REQUIRE(mesh.nBoundaryFaces() == 24);
        REQUIRE(mesh.nBoundaries() == 6);
        REQUIRE(mesh.nFaces() == 36);

        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.size() == 27);

        // Each cell is 0.5 * 0.5 * 0.5 = 0.125
        auto hostVols = mesh.cellVolumes().copyToHost();
        for (NeoN::localIdx c = 0; c < 8; ++c)
            REQUIRE(hostVols.view()[c] == Catch::Approx(0.125));

        // Cell centres at 0.25 increments
        auto hostCC = mesh.cellCentres().copyToHost();
        // cell(0,0,0) → (0.25, 0.25, 0.25)
        REQUIRE(hostCC.view()[0][0] == Catch::Approx(0.25));
        REQUIRE(hostCC.view()[0][1] == Catch::Approx(0.25));
        REQUIRE(hostCC.view()[0][2] == Catch::Approx(0.25));
        // cell(1,1,1) = k*nx*ny + j*nx + i = 1*4 + 1*2 + 1 = 7 → (0.75, 0.75, 0.75)
        REQUIRE(hostCC.view()[7][0] == Catch::Approx(0.75));
        REQUIRE(hostCC.view()[7][1] == Catch::Approx(0.75));
        REQUIRE(hostCC.view()[7][2] == Catch::Approx(0.75));

        // Boundary delta: left boundary first face should have negative x delta
        auto hostBndDelta = mesh.boundaryMesh().delta().copyToHost();
        REQUIRE(hostBndDelta.view()[0][0] == Catch::Approx(-0.25));
        REQUIRE(hostBndDelta.view()[0][1] == Catch::Approx(0.0));
        REQUIRE(hostBndDelta.view()[0][2] == Catch::Approx(0.0));
    }

    SECTION("Can create a uniform 3D grid with non-unit domain " + execName)
    {
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 2;
        NeoN::scalar Lx = 3.0;
        NeoN::scalar Ly = 2.0;
        NeoN::scalar Lz = 2.0;
        auto mesh = NeoN::createUniform3DGrid(exec, nx, ny, nz, Lx, Ly, Lz);

        // 3x2x2: 12 cells
        // internal: x:(3-1)*2*2=8, y:3*(2-1)*2=6, z:3*2*(2-1)=6 → 20
        // boundary: left(4)+right(4)+bottom(6)+top(6)+front(6)+back(6) = 32
        REQUIRE(mesh.nCells() == 12);
        REQUIRE(mesh.nInternalFaces() == 20);
        REQUIRE(mesh.nBoundaryFaces() == 32);
        REQUIRE(mesh.nBoundaries() == 6);
        REQUIRE(mesh.nFaces() == 52);

        // Cell volume = (3/3) * (2/2) * (2/2) = 1.0
        auto hostVols = mesh.cellVolumes().copyToHost();
        for (NeoN::localIdx c = 0; c < 12; ++c)
            REQUIRE(hostVols.view()[c] == Catch::Approx(1.0));
    }

    SECTION("3D grid patch face centres lie on correct planes (3x2x4) " + execName)
    {
        // Domain: [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
        // Boundary patches and the planes they must lie on:
        //   patch 0 "xmin"  — x = xmin plane  (ny*nz = 8  faces)
        //   patch 1 "xmax"  — x = xmax plane  (ny*nz = 8  faces)
        //   patch 2 "ymin"  — y = ymin plane  (nx*nz = 12 faces)
        //   patch 3 "ymax"  — y = ymax plane  (nx*nz = 12 faces)
        //   patch 4 "zmin"  — z = zmin plane  (nx*ny = 6  faces)
        //   patch 5 "zmax"  — z = zmax plane  (nx*ny = 6  faces)
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 4;
        NeoN::scalar xmin = 0.0;
        NeoN::scalar xmax = 3.0;
        NeoN::scalar ymin = 0.0;
        NeoN::scalar ymax = 2.0;
        NeoN::scalar zmin = 0.0;
        NeoN::scalar zmax = 4.0;
        auto mesh = NeoN::createUniform3DGrid(exec, nx, ny, nz, xmax, ymax, zmax);

        auto& bm = mesh.boundaryMesh();
        auto& offset = bm.offset();
        REQUIRE(offset.size() == 7);
        REQUIRE(offset[1] - offset[0] == 8);  // left
        REQUIRE(offset[2] - offset[1] == 8);  // right
        REQUIRE(offset[3] - offset[2] == 12); // bottom
        REQUIRE(offset[4] - offset[3] == 12); // top
        REQUIRE(offset[5] - offset[4] == 6);  // front
        REQUIRE(offset[6] - offset[5] == 6);  // back

        auto hostCf = bm.cf().copyToHost();

        // left (patch 0): all face centres on x = xmin plane
        for (NeoN::localIdx f = offset[0]; f < offset[1]; ++f)
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmin).margin(1e-10));

        // right (patch 1): all face centres on x = xmax plane
        for (NeoN::localIdx f = offset[1]; f < offset[2]; ++f)
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmax).margin(1e-10));

        // bottom (patch 2): all face centres on y = ymin plane
        for (NeoN::localIdx f = offset[2]; f < offset[3]; ++f)
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymin).margin(1e-10));

        // top (patch 3): all face centres on y = ymax plane
        for (NeoN::localIdx f = offset[3]; f < offset[4]; ++f)
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymax).margin(1e-10));

        // front (patch 4): all face centres on z = zmin plane
        for (NeoN::localIdx f = offset[4]; f < offset[5]; ++f)
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmin).margin(1e-10));

        // back (patch 5): all face centres on z = zmax plane
        for (NeoN::localIdx f = offset[5]; f < offset[6]; ++f)
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmax).margin(1e-10));
    }

    SECTION("Uniform 3D grid stores face node connectivity in stencilDB " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 2;
        auto mesh = NeoN::createUniform3DGrid(exec, nx, ny, nz);

        REQUIRE(mesh.stencilDB().contains("io::faceNodes"));

        auto& faceNodes =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<NeoN::localIdx>>>>(
                "io::faceNodes"
            );

        REQUIRE(static_cast<NeoN::localIdx>(faceNodes.size()) == mesh.nFaces());

        for (auto& fn : faceNodes)
            REQUIRE(fn.size() == 4);

        REQUIRE(mesh.stencilDB().contains("io::patchNames"));

        auto& patchNames =
            *mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");

        REQUIRE(patchNames.size() == 6);
        REQUIRE(patchNames[0] == "xmin");
        REQUIRE(patchNames[1] == "xmax");
        REQUIRE(patchNames[2] == "ymin");
        REQUIRE(patchNames[3] == "ymax");
        REQUIRE(patchNames[4] == "zmin");
        REQUIRE(patchNames[5] == "zmax");
    }
}
