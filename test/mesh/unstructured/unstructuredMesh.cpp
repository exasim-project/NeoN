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
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0

        NeoN::localIdx nCells = 4;

        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, nCells);

        // 1D mesh now delegates to create3DUniformMesh(exec, 4, 1, 1)
        // → hex slab topology with 2 patches
        // nInternalFaces: x:(4-1)*1*1=3, y:0, z:0 → 3
        // nBoundaryFaces: left(1)+right(1) = 2
        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nInternalFaces() == 3);
        REQUIRE(mesh.nBoundaryFaces() == 2);
        REQUIRE(mesh.nBoundaries() == 2);
        REQUIRE(mesh.nFaces() == 5);

        // Verify mesh points
        // bc  [   internal  ]  bc
        // 0.0 [ 0.25 | 0.50 ] 1.0
        // Verify point count: (4+1)*(1+1)*(1+1) = 20
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.size() == 20);

        // Verify cell centres at (i+0.5)*0.25, 0.5, 0.5
        auto cellCentresExp = std::vector<NeoN::Vec3> {
            {0.125, 0.5, 0.5}, {0.375, 0.5, 0.5}, {0.625, 0.5, 0.5}, {0.875, 0.5, 0.5}
        };
        REQUIRE_THAT(mesh.cellCentres(), Equals(cellCentresExp, ApproxVec3 {1e-12}));

        // Verify face owners (x-direction)
        auto faceOwnerExp = std::vector<NeoN::label> {0, 1, 2, 0, 3};
        REQUIRE_THAT(mesh.faceOwner(), Equals(faceOwnerExp, EqualInt()));

        // Verify face neighbours (x-direction)
        auto faceNeighbourExp = std::vector<NeoN::label> {1, 2, 3};
        REQUIRE_THAT(mesh.faceNeighbour(), Equals(faceNeighbourExp, EqualInt()));

        // Verify boundary mesh: first 2 boundary faces are xmin/xmax
        // Verify neighbouring cell for boundary faces
        auto faceCellsExp = std::vector<NeoN::localIdx> {0, 3};
        REQUIRE_THAT(mesh.boundaryMesh().faceCells(), Equals(faceCellsExp, EqualInt()));

        // // Verify neighbor cell centres
        auto cnExp = std::vector<NeoN::Vec3> {{0.125, 0.5, 0.5}, {0.875, 0.5, 0.5}};
        REQUIRE_THAT(mesh.boundaryMesh().cn(), Equals(cnExp, ApproxVec3 {1e-12}));

        // // Verify delta vectors
        auto deltaExp = std::vector<NeoN::Vec3> {{-0.125, 0.0, 0.0}, {0.125, 0.0, 0.0}};
        REQUIRE_THAT(mesh.boundaryMesh().delta(), Equals(deltaExp, ApproxVec3 {1e-12}));

        // Verify stencilDB has patchNames
        REQUIRE(mesh.stencilDB().contains(std::string("stencilPatchNames")));
    }

    SECTION("Can create a uniform 2D mesh" + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        auto mesh = NeoN::create2DUniformMesh(exec, nx, ny);

        // Topology: 2x2 mesh, one cell thick in z (hex cells)
        // - 4 hex cells
        // - 18 points = (2+1)*(2+1)*2  (two z-planes)
        // - 4 internal faces (quad): 2 vertical + 2 horizontal
        // - 8 boundary faces: left(2) + right(2) + bottom(2) + top(2)
        // - 4 boundaries: left, right, bottom, top
        // - 12 total faces
        REQUIRE(mesh.nCells() == 4);
        REQUIRE(mesh.nInternalFaces() == 4);
        REQUIRE(mesh.nBoundaryFaces() == 8);
        REQUIRE(mesh.nBoundaries() == 4);
        REQUIRE(mesh.nFaces() == 12);

        // Verify point count: two z-planes
        auto hostPoints = mesh.points().copyToHost();
        REQUIRE(hostPoints.size() == 18);

        // Verify cell volumes (each cell is 0.5 * 0.5 * 1.0 = 0.25)
        auto cellVolumesExp = {0.25, 0.25, 0.25, 0.25};
        REQUIRE_THAT(mesh.cellVolumes(), Equals(cellVolumesExp, ApproxScalar(1e-12)));

        // Verify cell centres (z should be 0.5, halfway through the slab)
        auto cellCentresExp = std::vector<NeoN::Vec3> {
            {0.25, 0.25, 0.5}, {0.75, 0.25, 0.5}, {0.25, 0.75, 0.5}, {0.75, 0.75, 0.5}
        };
        REQUIRE_THAT(mesh.cellCentres(), Equals(cellCentresExp, ApproxVec3 {1e-12}));

        // Verify the number of neighbouring cells for boundary faces
        // 4 patches: left(2), right(2), bottom(2), top(2)
        auto& bm = mesh.boundaryMesh();
        REQUIRE(bm.faceCells().size() == 8);

        // Verify boundary delta vectors
        // Left boundary: delta.x should be negative (face at x=0, cell centre at x=0.25)
        auto boundaryDeltaExp = std::vector<NeoN::Vec3> {
            {-0.25, 0.0, 0.0},
            {-0.25, 0.0, 0.0},
            {0.25, 0.0, 0.0},
            {0.25, 0.0, 0.0},
            {0.0, -0.25, 0.0},
            {0.0, -0.25, 0.0},
            {0.0, 0.25, 0.0},
            {0.0, 0.25, 0.0}
        };
        REQUIRE_THAT(bm.delta(), Equals(boundaryDeltaExp, ApproxVec3 {1e-12}));
    }

    SECTION("Uniform 2D mesh stores patch names in stencilDB " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        auto mesh = NeoN::create2DUniformMesh(exec, nx, ny);

        // stencilDB must contain std::string("stencilPatchNames")
        REQUIRE(mesh.stencilDB().contains(std::string("stencilPatchNames")));

        auto& patchNames = *mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>(
            std::string("stencilPatchNames")
        );

        // Must have one entry per boundary (4 patches)
        REQUIRE(patchNames.size() == 4);

        // Patch names must match the boundary order:
        // xmin, xmax, ymin, ymax
        REQUIRE(patchNames[0] == "xmin");
        REQUIRE(patchNames[1] == "xmax");
        REQUIRE(patchNames[2] == "ymin");
        REQUIRE(patchNames[3] == "ymax");
    }

    SECTION("Can create a uniform 2D mesh with non-unit domain " + execName)
    {
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::scalar lx = 3.0;
        NeoN::scalar ly = 2.0;
        auto mesh = NeoN::create2DUniformMesh(exec, nx, ny, lx, ly);

        // 3x2 mesh: 6 hex cells, (3-1)*2 + 3*(2-1) = 4+3 = 7 internal faces
        // boundary: left(2) + right(2) + bottom(3) + top(3) = 10
        REQUIRE(mesh.nCells() == 6);
        REQUIRE(mesh.nInternalFaces() == 7);
        REQUIRE(mesh.nBoundaryFaces() == 10);
        REQUIRE(mesh.nBoundaries() == 4);
        REQUIRE(mesh.nFaces() == 17);

        // Cell volume = (3/3) * (2/2) * 1.0 = 1.0
        auto cellVolumesExp = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        REQUIRE_THAT(mesh.cellVolumes(), Equals(cellVolumesExp, ApproxScalar(1e-12)));
    }

    SECTION("2D mesh patch face centres lie on correct planes (3x2) " + execName)
    {
        // Domain: [xmin, xmax] x [ymin, ymax] (one cell thick in z)
        // Boundary patches and the planes they must lie on:
        //   patch 0 "xmin"  — x = xmin plane  (ny    = 2 faces)
        //   patch 1 "xmax"  — x = xmax plane  (ny    = 2 faces)
        //   patch 2 "ymin"  — y = ymin plane  (nx    = 3 faces)
        //   patch 3 "ymax"  — y = ymax plane  (nx    = 3 faces)
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::scalar xmin = 0.0;
        NeoN::scalar xmax = 3.0;
        NeoN::scalar ymin = 0.0;
        NeoN::scalar ymax = 2.0;
        auto mesh = NeoN::create2DUniformMesh(exec, nx, ny, xmax, ymax);

        auto& bm = mesh.boundaryMesh();
        auto& offset = bm.offset();
        REQUIRE(offset.size() - 1 == 4);     // 4 patches
        REQUIRE(offset[1] - offset[0] == 2); // left
        REQUIRE(offset[2] - offset[1] == 2); // right
        REQUIRE(offset[3] - offset[2] == 3); // bottom
        REQUIRE(offset[4] - offset[3] == 3); // top

        // Verify face centres for each patch
        auto cfExp = std::vector<NeoN::Vec3> {
            {xmin, 0.5, 0.5},
            {xmin, 1.5, 0.5}, // left
            {xmax, 0.5, 0.5},
            {xmax, 1.5, 0.5}, // right
            {0.5, ymin, 0.5},
            {1.5, ymin, 0.5},
            {2.5, ymin, 0.5}, // bottom
            {0.5, ymax, 0.5},
            {1.5, ymax, 0.5},
            {2.5, ymax, 0.5} // top
        };
        REQUIRE_THAT(bm.cf(), Equals(cfExp, ApproxVec3 {1e-12}));
    }

    SECTION("Can create a uniform 3D mesh (2x2x2) " + execName)
    {
        NeoN::localIdx nx = 2;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 2;
        auto mesh = NeoN::create3DUniformMesh(exec, nx, ny, nz);

        // Topology: 2x2x2 mesh
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
        auto cellVolumesExp = {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125};
        REQUIRE_THAT(mesh.cellVolumes(), Equals(cellVolumesExp, ApproxScalar(1e-12)));

        // Cell centres at 0.25 increments
        // cell(0,0,0) → (0.25, 0.25, 0.25)
        // cell(1,1,1) = k*nx*ny + j*nx + i = 1*4 + 1*2 + 1 = 7 → (0.75, 0.75, 0.75)
        auto cellCentresExp = std::vector<NeoN::Vec3> {
            {0.25, 0.25, 0.25},
            {0.75, 0.25, 0.25},
            {0.25, 0.75, 0.25},
            {0.75, 0.75, 0.25},
            {0.25, 0.25, 0.75},
            {0.75, 0.25, 0.75},
            {0.25, 0.75, 0.75},
            {0.75, 0.75, 0.75}
        };
        REQUIRE_THAT(mesh.cellCentres(), Equals(cellCentresExp, ApproxVec3 {1e-12}));

        // Boundary delta: left boundary first face should have negative x delta
        auto hostBndDelta = mesh.boundaryMesh().delta().copyToHost();
        REQUIRE(hostBndDelta.view()[0][0] == Catch::Approx(-0.25));
        REQUIRE(hostBndDelta.view()[0][1] == Catch::Approx(0.0));
        REQUIRE(hostBndDelta.view()[0][2] == Catch::Approx(0.0));
    }

    SECTION("Can create a uniform 3D mesh with non-unit domain " + execName)
    {
        NeoN::localIdx nx = 3;
        NeoN::localIdx ny = 2;
        NeoN::localIdx nz = 2;
        NeoN::scalar lx = 3.0;
        NeoN::scalar ly = 2.0;
        NeoN::scalar lz = 2.0;
        auto mesh = NeoN::create3DUniformMesh(exec, nx, ny, nz, lx, ly, lz);

        // 3x2x2: 12 cells
        // internal: x:(3-1)*2*2=8, y:3*(2-1)*2=6, z:3*2*(2-1)=6 → 20
        // boundary: left(4)+right(4)+bottom(6)+top(6)+front(6)+back(6) = 32
        REQUIRE(mesh.nCells() == 12);
        REQUIRE(mesh.nInternalFaces() == 20);
        REQUIRE(mesh.nBoundaryFaces() == 32);
        REQUIRE(mesh.nBoundaries() == 6);
        REQUIRE(mesh.nFaces() == 52);

        // Cell volume = (3/3) * (2/2) * (2/2) = 1.0
        auto cellVolumesExp =
            std::vector<NeoN::scalar> {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        REQUIRE_THAT(mesh.cellVolumes(), Equals(cellVolumesExp, ApproxScalar(1e-12)));
    }

    SECTION("3D mesh patch face centres lie on correct planes (3x2x4) " + execName)
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
        auto mesh = NeoN::create3DUniformMesh(exec, nx, ny, nz, xmax, ymax, zmax);

        auto& bm = mesh.boundaryMesh();
        auto& offset = bm.offset();
        REQUIRE(offset.size() == 7);
        REQUIRE(offset[1] - offset[0] == 8);  // left
        REQUIRE(offset[2] - offset[1] == 8);  // right
        REQUIRE(offset[3] - offset[2] == 12); // bottom
        REQUIRE(offset[4] - offset[3] == 12); // top
        REQUIRE(offset[5] - offset[4] == 6);  // front
        REQUIRE(offset[6] - offset[5] == 6);  // back

        // Verify face centres for each patch
        auto hostCf = bm.cf().copyToHost();

        // left (patch 0): all face centres on x = xmin plane
        for (NeoN::localIdx f = offset[0]; f < offset[1]; ++f)
        {
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmin).margin(1e-10));
        }

        // right (patch 1): all face centres on x = xmax plane
        for (NeoN::localIdx f = offset[1]; f < offset[2]; ++f)
        {
            REQUIRE(hostCf.view()[f][0] == Catch::Approx(xmax).margin(1e-10));
        }

        // bottom (patch 2): all face centres on y = ymin plane
        for (NeoN::localIdx f = offset[2]; f < offset[3]; ++f)
        {
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymin).margin(1e-10));
        }

        // top (patch 3): all face centres on y = ymax plane
        for (NeoN::localIdx f = offset[3]; f < offset[4]; ++f)
        {
            REQUIRE(hostCf.view()[f][1] == Catch::Approx(ymax).margin(1e-10));
        }

        // front (patch 4): all face centres on z = zmin plane
        for (NeoN::localIdx f = offset[4]; f < offset[5]; ++f)
        {
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmin).margin(1e-10));
        }

        // back (patch 5): all face centres on z = zmax plane
        for (NeoN::localIdx f = offset[5]; f < offset[6]; ++f)
        {
            REQUIRE(hostCf.view()[f][2] == Catch::Approx(zmax).margin(1e-10));
        }
    }
}
