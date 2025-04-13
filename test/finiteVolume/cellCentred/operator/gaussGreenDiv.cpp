// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    auto boundFaceFlux = faceFlux.internalVector().view();
    // face on the left side has different orientation
    parallelFor(
        exec,
        {mesh.nCells() - 1, mesh.nCells()},
        KOKKOS_LAMBDA(const size_t i) { boundFaceFlux[i] = -1.0; }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), one<TestType>());
    fill(phi.boundaryVector().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Vector<TestType>(exec, phi.size());
    fill(result, zero<TestType>());

    SECTION("Construct from Token" + execName)
    {
        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        Input input = Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
        op.div(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == zero<TestType>());
        }
    }
}

}
