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

    // Build boundary values for all boundary faces
    // xmin(1): value=0.5, xmax(1): value=10.5
    // y/z faces: value = owning cell value (no gradient in y/z)
    {
        auto& bm = mesh.boundaryMesh();
        auto& offset = bm.offset();
        auto hostFaceCells = bm.faceCells().copyToHost();
        auto nBnd = mesh.nBoundaryFaces();
        std::vector<TestType> bndVals(static_cast<size_t>(nBnd));
        // xmin face
        bndVals[0] = 0.5 * one<TestType>();
        // xmax face
        bndVals[1] = 10.5 * one<TestType>();
        // y/z faces: boundary value = cell value = (cellId + 1)
        for (NeoN::localIdx f = offset[2]; f < nBnd; ++f)
        {
            auto cellId = hostFaceCells.view()[f];
            bndVals[static_cast<size_t>(f)] = scalar(cellId + 1) * one<TestType>();
        }
        phi.boundaryData().value() = NeoN::Vector<TestType>(exec, bndVals);
    }

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

        auto phifHost = phif.internalVector().copyToHost();
        auto sPhif = phifHost.view();
        auto nI = mesh.nInternalFaces();
        // Internal x-faces: gradient = 10.0
        for (NeoN::localIdx i = 0; i < nI; i++)
        {
            REQUIRE(
                NeoN::mag(sPhif[i] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
            );
        }
        // xmin boundary face (index nI): gradient = -10.0
        REQUIRE(NeoN::mag(sPhif[nI] + 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8));
        // xmax boundary face (index nI+1): gradient = 10.0
        REQUIRE(
            NeoN::mag(sPhif[nI + 1] - 10.0 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8)
        );
        // y/z boundary faces: gradient = 0 (boundary value == cell value)
        for (NeoN::localIdx i = nI + 2; i < nI + mesh.nBoundaryFaces(); i++)
        {
            REQUIRE(NeoN::mag(sPhif[i]) == Catch::Approx(0.0).margin(1e-8));
        }
    }
}
}
