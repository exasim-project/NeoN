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

    SECTION("Can assemble linear system from expression on " + execName)
    {
        auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
        fvcc::VolumeField<scalar> phi(exec, "sf", mesh, volumeBCs);
        fill(phi.internalVector(), 1.0);
        fill(phi.boundaryData().value(), 1.0);
        phi.correctBoundaryConditions();

        fvcc::VolumeField<scalar> rhs(exec, "sf", mesh, volumeBCs);
        rhs.correctBoundaryConditions();
        fill(rhs.internalVector(), 1.0);

        auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
        fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
        fill(faceFlux.internalVector(), 1.0);
        auto boundFaceFlux = faceFlux.internalVector().view();
        // face on the left side has different orientation
        parallelFor(
            exec,
            {mesh.nCells() - 1, mesh.nCells()},
            KOKKOS_LAMBDA(const localIdx i) { boundFaceFlux[i] = -1.0; }
        );

        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        auto div = fvcc::DivOperator(NeoN::dsl::Operator::Type::Implicit, faceFlux, phi, input);

        auto expr = NeoN::dsl::Expression<NeoN::scalar>(exec);
        expr.addOperator(div);

        auto schemesDict = NeoN::Dictionary();

        auto ls = assembleLinearSystem<NeoN::scalar>(expr, rhs, schemesDict, 0, 0.1);

        auto lsHost = ls.copyToHost();
        auto [csrHV, rhsHV] = lsHost.view();

        // All values should be zero since original field was uniform
        for (NeoN::localIdx i = 0; i < csrHV.values.size(); ++i)
        {
            std::cout << " csr[" << i << "]:" << csrHV.values[i] << "\n";
        }
        // for (NeoN::localIdx i = 0; i < csrHV.values.size(); ++i)
        // {
        //     REQUIRE(csrHV.values[i] == 0.0);
        // }

        for (NeoN::localIdx i = 0; i < rhsHV.size(); ++i)
        {
            REQUIRE(rhsHV[i] == 0.0);
        }
    }
}

}
