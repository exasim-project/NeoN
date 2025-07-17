// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("volumeVector")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
    std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
    for (auto patchi : I<NeoN::localIdx> {0, 1, 2, 3})
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
    }

    SECTION("can instantiate volumeVector with fixedValues on: " + execName)
    {
        fvcc::VolumeField<NeoN::scalar> vf(exec, "vf", mesh, bcs);
        NeoN::fill(vf.internalVector(), 1.0);
        vf.correctBoundaryConditions();

        NeoN::Vector<NeoN::scalar> internalVector(mesh.exec(), mesh.nCells(), 1.0);

        REQUIRE(vf.exec() == exec);
        REQUIRE(vf.internalVector().size() == 1);

        auto internalValues = vf.internalVector().copyToHost();
        for (NeoN::localIdx i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues.view()[i] == 1.0);
        }

        auto values = vf.boundaryData().value().copyToHost();

        for (NeoN::localIdx i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == 2.0);
        }

        auto refValue = vf.boundaryData().refValue().copyToHost();
        for (NeoN::localIdx i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.view()[i] == 2.0);
        }
    }

    SECTION("can instantiate volumeVector with fixedValues from internal Vector on: " + execName)
    {
        NeoN::Vector<NeoN::scalar> internalVector(mesh.exec(), mesh.nCells(), 1.0);

        fvcc::VolumeField<NeoN::scalar> vf(exec, "vf", mesh, internalVector, bcs);
        vf.correctBoundaryConditions();

        auto internalValues = vf.internalVector().copyToHost();
        for (NeoN::localIdx i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues.view()[i] == 1.0);
        }

        auto values = vf.boundaryData().value().copyToHost();

        for (NeoN::localIdx i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == 2.0);
        }

        auto refValue = vf.boundaryData().refValue().copyToHost();
        for (NeoN::localIdx i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.view()[i] == 2.0);
        }
    }
}
