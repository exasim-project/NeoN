// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("Courant Number")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("can determine maximum courant number of flux field on 1D uniform mesh: " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, 4);
        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcs {};
        // xmin/xmax patches: fixedValue = 1.0
        for (NeoN::localIdx patchi = 0; patchi < 2; ++patchi)
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 1.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }
        // y/z patches: fixedValue = 0.0 (not relevant for 1D flow)
        for (NeoN::localIdx patchi = 2; patchi < mesh.nBoundaries(); ++patchi)
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 0.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> sf(exec, "sf", mesh, bcs);
        // Only x-direction internal faces carry flux
        NeoN::fill(sf.internalVector(), 0.0);
        auto sfView = sf.internalVector().view();
        auto nI = mesh.nInternalFaces();
        NeoN::parallelFor(
            exec, {0, nI}, NEON_LAMBDA(const NeoN::localIdx i) { sfView[i] = 1.0; }
        );
        sf.correctBoundaryConditions();

        // use arbitrary time step size of 0.01
        // For (4,1,1) mesh with x-only flux=1.0: each cell sees 2 x-faces
        // sum|flux| = 2.0, cellVol = 0.25
        // CoNum = 0.5 * 2.0/0.25 * 0.01 = 0.04
        const auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(sf, 0.01);

        REQUIRE(maxCoNum == 0.04);
    }
}
