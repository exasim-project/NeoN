// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGradVec3.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

TEST_CASE("GaussGreenGradVec3")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(mesh);
    fvcc::VolumeField<Vec3> phi(exec, "phi", mesh, volumeBCs);
    fill(phi.internalVector(), one<Vec3>());
    fill(phi.boundaryData().value(), one<Vec3>());
    phi.correctBoundaryConditions();

    SECTION("Gradient of uniform Vec3 field is zero tensor" + execName)
    {
        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        auto gradOp = fvcc::GradOperator<Tensor>(Operator::Type::Explicit, phi, input);

        auto result = Vector<Tensor>(exec, phi.size());
        fill(result, zero<Tensor>());
        gradOp.explicitOperation(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            for (size_t c = 0; c < 9; c++)
            {
                REQUIRE(outHostView[i][c] == Catch::Approx(0.0).margin(1e-10));
            }
        }
    }
}

}
