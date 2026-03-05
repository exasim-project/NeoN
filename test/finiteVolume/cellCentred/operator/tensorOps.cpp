// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/tensorOps.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

TEST_CASE("TensorFieldOps")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto tensorBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Tensor>>(mesh);

    // Create a tensor field with known values: T = (1,2,3,4,5,6,7,8,9)
    fvcc::VolumeField<Tensor> T(exec, "T", mesh, tensorBCs);
    fill(T.internalVector(), Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    fill(T.boundaryData().value(), Tensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    T.correctBoundaryConditions();

    SECTION("symm(TensorField) -> SymmTensorField" + execName)
    {
        auto result = fvcc::symm(T);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        // symm(T) = 0.5*(T + T^T)
        // xx=1, xy=0.5*(2+4)=3, xz=0.5*(3+7)=5, yy=5, yz=0.5*(6+8)=7, zz=9
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i].xx() == Catch::Approx(1.0));
            REQUIRE(outHostView[i].xy() == Catch::Approx(3.0));
            REQUIRE(outHostView[i].xz() == Catch::Approx(5.0));
            REQUIRE(outHostView[i].yy() == Catch::Approx(5.0));
            REQUIRE(outHostView[i].yz() == Catch::Approx(7.0));
            REQUIRE(outHostView[i].zz() == Catch::Approx(9.0));
        }
    }

    SECTION("skew(TensorField) -> TensorField" + execName)
    {
        auto result = fvcc::skew(T);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        // skew(T) = 0.5*(T - T^T)
        // diagonal should be zero
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i].xx() == Catch::Approx(0.0));
            REQUIRE(outHostView[i].yy() == Catch::Approx(0.0));
            REQUIRE(outHostView[i].zz() == Catch::Approx(0.0));
        }
    }

    SECTION("twoSymm(TensorField) -> SymmTensorField" + execName)
    {
        auto result = fvcc::twoSymm(T);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        // twoSymm(T) = T + T^T
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i].xx() == Catch::Approx(2.0));
            REQUIRE(outHostView[i].xy() == Catch::Approx(6.0));
            REQUIRE(outHostView[i].xz() == Catch::Approx(10.0));
            REQUIRE(outHostView[i].yy() == Catch::Approx(10.0));
            REQUIRE(outHostView[i].yz() == Catch::Approx(14.0));
            REQUIRE(outHostView[i].zz() == Catch::Approx(18.0));
        }
    }

    SECTION("mag(TensorField) -> ScalarField" + execName)
    {
        auto result = fvcc::mag(T);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        // mag = sqrt(sum of squares) = sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285)
        scalar expected = std::sqrt(285.0);
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == Catch::Approx(expected));
        }
    }

    SECTION("mag(SymmTensorField) -> ScalarField" + execName)
    {
        auto symmBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<SymmTensor>>(mesh);
        fvcc::VolumeField<SymmTensor> S(exec, "S", mesh, symmBCs);
        fill(S.internalVector(), SymmTensor(1.0, 0.0, 0.0, 1.0, 0.0, 1.0));
        S.correctBoundaryConditions();

        auto result = fvcc::mag(S);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == Catch::Approx(std::sqrt(3.0)));
        }
    }

    SECTION("dev(SymmTensorField) -> SymmTensorField" + execName)
    {
        auto symmBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<SymmTensor>>(mesh);
        fvcc::VolumeField<SymmTensor> S(exec, "S", mesh, symmBCs);
        // identity SymmTensor
        fill(S.internalVector(), SymmTensor(1.0, 0.0, 0.0, 1.0, 0.0, 1.0));
        S.correctBoundaryConditions();

        auto result = fvcc::dev(S);
        auto outHost = result.internalVector().copyToHost();
        auto outHostView = outHost.view();
        // dev(I) = I - tr(I)/3 * I = I - I = 0
        for (int i = 0; i < result.size(); i++)
        {
            for (size_t c = 0; c < 6; c++)
            {
                REQUIRE(outHostView[i][c] == Catch::Approx(0.0).margin(1e-10));
            }
        }
    }
}

}
