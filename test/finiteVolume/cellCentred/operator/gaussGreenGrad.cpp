// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// The Gauss-Green gradient is assembled cell-based (each cell gathers its own internal-face
// contributions) so that the owner/neighbour scatter needs no atomics. These tests pin the two
// properties that assembly must preserve: the owner/neighbour sign cancellation and exactness for
// linear fields.

TEST_CASE("GaussGreenGrad uniform field gives zero gradient")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    fill(phi.internalVector(), 1.0);
    fill(phi.boundaryData().value(), 1.0);
    phi.correctBoundaryConditions();

    fvcc::GaussGreenGrad gradOp(exec, mesh);

    SECTION("zero gradient " + execName)
    {
        // sum_f S_f over a closed cell is zero, so a uniform field must yield a zero gradient in
        // every cell regardless of the owner/neighbour sign convention.
        auto gradPhi = gradOp.grad(phi);

        auto gradHost = gradPhi.internalVector().copyToHost();
        auto gradView = gradHost.view();
        for (localIdx i = 0; i < gradView.size(); ++i)
        {
            REQUIRE(gradView[i][0] == Catch::Approx(0.0).margin(1e-12));
            REQUIRE(gradView[i][1] == Catch::Approx(0.0).margin(1e-12));
            REQUIRE(gradView[i][2] == Catch::Approx(0.0).margin(1e-12));
        }
    }
}

TEST_CASE("GaussGreenGrad reproduces the exact gradient of a linear field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    const scalar lx = 1.0;
    auto mesh = create1DUniformMesh(exec, nCells, lx);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);

    // phi = x (a linear field). Gauss-Green is exact for linear fields, so d(phi)/dx = 1 in every
    // interior cell (cells whose faces are all internal).
    auto phiView = phi.internalVector().view();
    const auto cc = mesh.cellCenters().view();
    parallelFor(
        exec, {0, nCells}, NEON_LAMBDA(const localIdx i) { phiView[i] = cc[i][0]; }, "setLinearPhi"
    );
    phi.correctBoundaryConditions();

    fvcc::GaussGreenGrad gradOp(exec, mesh);

    SECTION("interior gradient is exact " + execName)
    {
        auto gradPhi = gradOp.grad(phi);

        auto gradHost = gradPhi.internalVector().copyToHost();
        auto gradView = gradHost.view();
        // Skip the two boundary cells (index 0 and nCells-1): their gradient also depends on the
        // boundary face value, which is not part of the cell-based internal assembly under test.
        for (localIdx i = 1; i < nCells - 1; ++i)
        {
            REQUIRE(gradView[i][0] == Catch::Approx(1.0).margin(1e-12));
            REQUIRE(gradView[i][1] == Catch::Approx(0.0).margin(1e-12));
            REQUIRE(gradView[i][2] == Catch::Approx(0.0).margin(1e-12));
        }
    }
}

} // namespace NeoN
