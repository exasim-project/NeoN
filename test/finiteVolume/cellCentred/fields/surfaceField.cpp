// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("surfaceField")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("can instantiate SurfaceField with fixedValues on: " + execName)
    {
        NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcs {};
        for (auto patchi : I<NeoN::size_t> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(mesh, dict, patchi));
        }

        fvcc::SurfaceField<NeoN::scalar> sf(exec, "sf", mesh, bcs);
        // the internal field is 4 because the mesh has 4 boundaryFaces
        NeoN::fill(sf.internalField(), 1.0);

        {
            auto internalValues = sf.internalField().copyToHost();
            for (size_t i = 0; i < internalValues.size(); ++i)
            {
                REQUIRE(internalValues.view()[i] == 1.0);
            }
        }

        sf.correctBoundaryConditions();
        REQUIRE(sf.exec() == exec);
        REQUIRE(sf.internalField().size() == 4);

        // the correctBoundaryConditions also sets the internalfield to 2.0
        // for surfaceFields
        auto internalValues = sf.internalField().copyToHost();
        for (size_t i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues.view()[i] == 2.0);
        }

        auto values = sf.boundaryField().value().copyToHost();
        for (size_t i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == 2.0);
        }

        auto refValue = sf.boundaryField().refValue().copyToHost();
        for (size_t i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.view()[i] == 2.0);
        }
    }
}
