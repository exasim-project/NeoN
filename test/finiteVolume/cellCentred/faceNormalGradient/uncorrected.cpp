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

TEMPLATE_TEST_CASE("uncorrected", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = 10;
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

    SECTION("Construct from Token" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
    }

    SECTION("faceNormalGrad" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, input);
        uncorrected.faceNormalGrad(phi, phif);

        // internal faces
        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        for (NeoN::localIdx i = 0; i < nCells - 1; i++)
        {
            // correct value is 10.0
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
}
}
