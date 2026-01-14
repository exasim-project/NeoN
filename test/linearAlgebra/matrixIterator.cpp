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

TEST_CASE("MatrixIterator")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;

    auto mesh = create1DUniformMesh(exec, nCells);
    auto mi = NeoN::la::createSparsityPatternMatrixIterator<NeoN::localIdx>(mesh);

    SECTION("Can construct sparsity pattern " + execName)
    {
        // some basic sanity checks
        REQUIRE(mi.diagOffset().size() == nCells);
        REQUIRE(mi.ownerOffset().size() == nFaces);
        REQUIRE(mi.neighbourOffset().size() == nFaces);
    }

    SECTION("has correct diagOffs" + execName)
    {
        auto diagOffs = mi.diagOffset().copyToHost();
        auto diagOffsS = diagOffs.view();

        REQUIRE(diagOffsS[0] == 0);
        REQUIRE(diagOffsS[1] == 1);
        REQUIRE(diagOffsS[2] == 1);
        REQUIRE(diagOffsS[3] == 1);
        REQUIRE(diagOffsS[4] == 1);
        REQUIRE(diagOffsS[5] == 1);
        REQUIRE(diagOffsS[6] == 1);
        REQUIRE(diagOffsS[7] == 1);
        REQUIRE(diagOffsS[8] == 1);
        REQUIRE(diagOffsS[9] == 1);
    }
}

}
