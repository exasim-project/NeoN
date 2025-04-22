// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

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
        for (auto patchi : I<NeoN::localIdx> {0, 1})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 1.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> sf(exec, "sf", mesh, bcs);
        NeoN::fill(sf.internalVector(), 1.0);
        sf.correctBoundaryConditions();

        // use arbitrary time step size of 0.01
        const NeoN::scalar coNum = fvcc::computeCoNum(sf, 0.01);

        REQUIRE(coNum == 0.04);
    }
}
