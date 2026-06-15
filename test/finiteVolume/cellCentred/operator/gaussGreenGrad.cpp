// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

TEMPLATE_TEST_CASE("Face based and cell based grad give same results", "[template]", NeoN::scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    fvcc::GaussGreenGrad gradOp(exec, mesh);

    SECTION("direct Vector overloads" + execName)
    {
        Vector<Vec3> gradFaceBased(exec, mesh.nCells(), zero<Vec3>());
        Vector<Vec3> gradCellBased(exec, mesh.nCells(), zero<Vec3>());

        gradOp.grad(phi, dsl::Coeff {1.0}, gradFaceBased);
        gradOp.gradCellBased(phi, dsl::Coeff {1.0}, gradCellBased);

        REQUIRE_THAT(gradFaceBased, Equals(gradCellBased, Approx {1e-12}));
    }

    SECTION("LinearSystem dispatch" + execName)
    {
        auto lsFaceBased = la::createEmptyLinearSystem<Vec3>(mesh);

        auto cellIterator = std::make_shared<la::CellBasedIterator>();
        auto lsCellBased = la::createEmptyLinearSystem<Vec3>(mesh, cellIterator);

        gradOp.grad(phi, dsl::Coeff {1.0}, lsFaceBased);
        gradOp.grad(phi, dsl::Coeff {1.0}, lsCellBased);

        REQUIRE_THAT(lsFaceBased.rhs(), Equals(lsCellBased.rhs(), Approx {1e-12}));
    }
}

} // namespace NeoN
