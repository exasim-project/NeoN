// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using SparsityPattern = fvcc::SparsityPattern;

namespace NeoN
{

TEST_CASE("SparsityPattern")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto sp = SparsityPattern {mesh};

    SECTION("Can construct sparsity pattern " + execName)
    {
        // some basic sanity checks
        REQUIRE(sp.ownerOffset().size() == nFaces);
        REQUIRE(sp.neighbourOffset().size() == nFaces);
        REQUIRE(sp.diagOffset().size() == nCells);
    }

    SECTION("has correct diagOffs" + execName)
    {
        auto diagOffs = sp.diagOffset().copyToHost();
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

    SECTION("Can produce rowPtrs " + execName)
    {
        auto rowPtr = sp.rowPtrs().copyToHost();
        auto rowPtrH = rowPtr.view();

        REQUIRE(rowPtrH[0] == 0);
        REQUIRE(rowPtrH[1] == 2);
        REQUIRE(rowPtrH[2] == 5);
        REQUIRE(rowPtrH[3] == 8);
        REQUIRE(rowPtrH[4] == 11);
        REQUIRE(rowPtrH[5] == 14);
        REQUIRE(rowPtrH[6] == 17);
        REQUIRE(rowPtrH[7] == 20);
        REQUIRE(rowPtrH[8] == 23);
        REQUIRE(rowPtrH[9] == 26);
        REQUIRE(rowPtrH[10] == 28);
    }

    SECTION("Can produce column indices " + execName)
    {
        auto colIdx = sp.colIdxs();
        auto colIdxH = colIdx.copyToHost();
        auto colIdxHS = colIdxH.view();

        REQUIRE(colIdx.size() == nCells + 2 * nFaces);
        REQUIRE(colIdxHS[0] == 0);
        REQUIRE(colIdxHS[1] == 1);

        REQUIRE(colIdxHS[2] == 0);
        REQUIRE(colIdxHS[3] == 1);
        REQUIRE(colIdxHS[4] == 2);

        REQUIRE(colIdxHS[5] == 1);
        REQUIRE(colIdxHS[6] == 2);
        REQUIRE(colIdxHS[7] == 3);

        REQUIRE(colIdxHS[8] == 2);
        REQUIRE(colIdxHS[9] == 3);
        REQUIRE(colIdxHS[10] == 4);
    }
}

}
