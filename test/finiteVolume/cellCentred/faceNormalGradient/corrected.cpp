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

TEMPLATE_TEST_CASE("corrected", "[template]", NeoN::scalar, NeoN::Vec3)
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
        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradient<TestType> corrected(exec, mesh, input);
        REQUIRE(corrected.corrected() == true);
    }

    SECTION("faceNormalGrad on orthogonal mesh matches uncorrected" + execName)
    {
        NeoN::Input inputCorr = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradient<TestType> corrected(exec, mesh, inputCorr);

        NeoN::Input inputUncorr = NeoN::TokenList({std::string("uncorrected")});
        fvcc::FaceNormalGradient<TestType> uncorrected(exec, mesh, inputUncorr);

        fvcc::SurfaceField<TestType> phifCorr(exec, "phifCorr", mesh, surfaceBCs);
        fill(phifCorr.internalVector(), zero<TestType>());
        corrected.faceNormalGrad(phi, phifCorr);

        fvcc::SurfaceField<TestType> phifUncorr(exec, "phifUncorr", mesh, surfaceBCs);
        fill(phifUncorr.internalVector(), zero<TestType>());
        uncorrected.faceNormalGrad(phi, phifUncorr);

        auto corrHost = phifCorr.internalVector().copyToHost();
        auto uncorrHost = phifUncorr.internalVector().copyToHost();
        auto sCorr = corrHost.view();
        auto sUncorr = uncorrHost.view();

        // On an orthogonal mesh, correction vectors are zero,
        // so corrected and uncorrected should give the same result
        for (NeoN::localIdx i = 0; i < static_cast<NeoN::localIdx>(sCorr.size()); i++)
        {
            REQUIRE(
                NeoN::mag(sCorr[i] - sUncorr[i]) == Catch::Approx(0.0).margin(1e-8)
            );
        }
    }

    SECTION("faceNormalGrad values" + execName)
    {
        NeoN::Input input = NeoN::TokenList({std::string("corrected")});
        fvcc::FaceNormalGradient<TestType> corrected(exec, mesh, input);
        corrected.faceNormalGrad(phi, phif);

        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        for (NeoN::localIdx i = 0; i < nCells - 1; i++)
        {
            // correct value is 10.0 (same as uncorrected on orthogonal mesh)
            REQUIRE(
                NeoN::mag(sPhif[i] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
            );
        }
    }
}
}
