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

TEST_CASE("SparsityPattern")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;

    auto mesh = create1DUniformMesh(exec, nCells);
    auto [sp, mi] =
        NeoN::la::createSparsityPatternMatrixIterator<NeoN::scalar, NeoN::localIdx>(mesh);

    SECTION("Can produce rowOffs " + execName)
    {
        auto rowPtr = sp->rowOffs().copyToHost();
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
        auto colIdx = sp->colIdxs();
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
