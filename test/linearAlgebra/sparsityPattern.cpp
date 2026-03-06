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
    auto mi = NeoN::la::createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(mesh);
    auto sp = mi->sparsityPattern();

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

    SECTION("Can generate views " + execName)
    {
        auto spH = sp->copyToHost();
        auto [colIdxs, rowOffs] = spH.view();

        REQUIRE(colIdxs[0] == 0);
        REQUIRE(colIdxs[1] == 1);

        REQUIRE(colIdxs[2] == 0);
        REQUIRE(colIdxs[3] == 1);
        REQUIRE(colIdxs[4] == 2);

        REQUIRE(colIdxs[5] == 1);
        REQUIRE(colIdxs[6] == 2);
        REQUIRE(colIdxs[7] == 3);

        REQUIRE(colIdxs[8] == 2);
        REQUIRE(colIdxs[9] == 3);
        REQUIRE(colIdxs[10] == 4);

        REQUIRE(rowOffs[0] == 0);
        REQUIRE(rowOffs[1] == 2);
        REQUIRE(rowOffs[2] == 5);
        REQUIRE(rowOffs[3] == 8);
        REQUIRE(rowOffs[4] == 11);
        REQUIRE(rowOffs[5] == 14);
        REQUIRE(rowOffs[6] == 17);
        REQUIRE(rowOffs[7] == 20);
        REQUIRE(rowOffs[8] == 23);
        REQUIRE(rowOffs[9] == 26);
        REQUIRE(rowOffs[10] == 28);
    }
}

}
