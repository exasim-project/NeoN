// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("surfaceVector")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("can instantiate SurfaceField with fixedValues on: " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcs {};
        for (auto patchi : I<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> sf(exec, "sf", mesh, bcs);
        // single-cell mesh has 0 internal faces and 4 boundary faces
        REQUIRE(sf.internalVector().size() == 0);
        REQUIRE(sf.boundaryData().value().size() == 4);

        // Fill boundary values with 1.0 before correctBoundaryConditions
        NeoN::fill(sf.boundaryData().value(), 1.0);
        {
            auto bValues = sf.boundaryData().value().copyToHost();
            for (NeoN::localIdx i = 0; i < bValues.size(); ++i)
            {
                REQUIRE(bValues.view()[i] == 1.0);
            }
        }

        sf.correctBoundaryConditions();
        REQUIRE(sf.exec() == exec);

        // correctBoundaryConditions sets boundary data to fixedValue (2.0)
        auto values = sf.boundaryData().value().copyToHost();
        for (NeoN::localIdx i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == 2.0);
        }

        auto refValue = sf.boundaryData().refValue().copyToHost();
        for (NeoN::localIdx i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.view()[i] == 2.0);
        }
    }
}
