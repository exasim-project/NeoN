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

template<typename T>
using I = std::initializer_list<T>;

TEMPLATE_TEST_CASE("limitedCorrected", "[template]", NeoN::scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
    // 1D uniform mesh is orthogonal, so corrVec = 0 everywhere
    // LimitedCorrected result should be identical to uncorrected on this mesh
    auto mesh = create1DUniformMesh(exec, nCells);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh);

    fvcc::SurfaceField<TestType> phif(exec, "phif", mesh, surfaceBCs);
    fill(phif.internalVector(), zero<TestType>());

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "phi", mesh, volumeBCs);
    NeoN::parallelFor(
        phi.internalVector(),
        NEON_LAMBDA(const NeoN::localIdx i) { return scalar(i + 1) * one<TestType>(); }
    );
    phi.boundaryData().value() =
        NeoN::Vector<TestType>(exec, {0.5 * one<TestType>(), 10.5 * one<TestType>()});

    SECTION("Construct from TokenList — terse OF form 'limited <coeff>'" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradient<TestType> lc(exec, mesh, input);
    }

    SECTION("Construct from TokenList — verbose OF form 'limited corrected <coeff>'" + execName)
    {
        NeoN::Input input =
            NeoN::TokenList({std::string("limited"), std::string("corrected"), NeoN::scalar(0.333)}
            );
        fvcc::FaceNormalGradient<TestType> lc(exec, mesh, input);
    }

    SECTION("Construct from Dictionary with limitCoeff" + execName)
    {
        NeoN::Input input = NeoN::Dictionary(
            {{"faceNormalGradient", std::string("limited")}, {"limitCoeff", NeoN::scalar(0.333)}}
        );
        fvcc::FaceNormalGradient<TestType> lc(exec, mesh, input);
    }

    SECTION("faceNormalGrad equals uncorrected on orthogonal mesh" + execName)
    {
        // On orthogonal meshes corrVec = 0, so limitedCorrected == uncorrected
        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(0.333)});
        fvcc::FaceNormalGradient<TestType> lc(exec, mesh, input);
        lc.faceNormalGrad(phi, phif);

        // internal faces
        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        for (NeoN::localIdx i = 0; i < nCells - 1; i++)
        {
            REQUIRE(
                NeoN::mag(sPhif[i] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
            );
        }
        // boundary faces are now in boundaryData().value()
        auto phifBHost = phif.boundaryData().value().copyToHost();
        auto sPhifB = phifBHost.view();
        // left boundary (bfi=0): gradient is -10.0
        REQUIRE(NeoN::mag(sPhifB[0] + 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8));
        // right boundary (bfi=1): gradient is 10.0
        REQUIRE(NeoN::mag(sPhifB[1] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8));
    }

    SECTION("faceNormalGrad with full correction (limitCoeff=1) on orthogonal mesh" + execName)
    {
        // limitCoeff=1 behaves as full corrected — still equals uncorrected on orthogonal mesh
        NeoN::Input input = NeoN::TokenList({std::string("limited"), NeoN::scalar(1.0)});
        fvcc::FaceNormalGradient<TestType> lc(exec, mesh, input);
        lc.faceNormalGrad(phi, phif);

        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        for (NeoN::localIdx i = 0; i < nCells - 1; i++)
        {
            REQUIRE(
                NeoN::mag(sPhif[i] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
            );
        }
    }
}
}
