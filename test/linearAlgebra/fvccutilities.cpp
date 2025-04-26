// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using SparsityPattern = fvcc::SparsityPattern;

namespace NeoN
{

TEST_CASE("Utilities")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
    auto mesh = create1DUniformMesh(exec, nCells);

    SECTION("Can assemble sparsity pattern from expression on " + execName)
    {
        auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
        fvcc::VolumeField<scalar> phi(exec, "sf", mesh, volumeBCs);
        phi.correctBoundaryConditions();

        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
        fill(faceFlux.internalVector(), 1.0);

        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        auto div = fvcc::DivOperator(NeoN::dsl::Operator::Type::Explicit, faceFlux, phi, input);
    }
}

}
