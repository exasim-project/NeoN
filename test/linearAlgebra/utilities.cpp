// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::Vec3;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;

TEST_CASE("Utilities")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Dense matrix
    // [ 1 2 3 ]   [1]   [2]   [6]     [2]
    // [ 4 5 6 ] x [1] - [2] = [15]  - [2]
    // [ 7 8 9 ]   [1]   [2]   [24]    [2]
    Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Vector<Vec3> mtxValues(
        exec,
        {{1.0, 1.0, 1.0},
         {2.0, 2.0, 2.0},
         {3.0, 3.0, 3.0},
         {4.0, 4.0, 4.0},
         {5.0, 5.0, 5.0},
         {6.0, 6.0, 6.0},
         {7.0, 7.0, 7.0},
         {8.0, 8.0, 8.0},
         {9.0, 9.0, 9.0}}
    );
    Vector<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs);

    // Sparse matrix
    // [ 1 2 0 ]
    // [ 4 5 6 ]
    // [ 0 8 9 ]
    Vector<scalar> valuesS(exec, {1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0});
    Vector<Vec3> mtxValuesS(
        exec,
        {{1.0, 1.0, 1.0},
         {2.0, 2.0, 2.0},
         {4.0, 4.0, 4.0},
         {5.0, 5.0, 5.0},
         {6.0, 6.0, 6.0},
         {8.0, 8.0, 8.0},
         {9.0, 9.0, 9.0}}
    );
    Vector<localIdx> colIdxS(exec, {0, 1, 0, 1, 2, 1, 2});
    Vector<localIdx> rowOffsS(exec, {0, 2, 5, 7});
    // CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs);
    SECTION("Can stretchRowPtrs " + execName)
    {
        auto res = NeoN::la::stretchRowPtrs(rowOffs);
        auto resHost = res.copyToHost();

        REQUIRE(resHost.view()[0] == 0);
        REQUIRE(resHost.view()[1] == 3);
        REQUIRE(resHost.view()[2] == 6);
        REQUIRE(resHost.view()[3] == 9);
        REQUIRE(resHost.view()[4] == 12);
        REQUIRE(resHost.view()[5] == 15);
    }

    SECTION("Can stretchRowPtrs2 " + execName)
    {
        auto res = NeoN::la::stretchRowPtrs(rowOffsS);
        auto resHost = res.copyToHost();

        REQUIRE(resHost.view()[0] == 0);
        REQUIRE(resHost.view()[1] == 2);
        REQUIRE(resHost.view()[2] == 4);
        REQUIRE(resHost.view()[3] == 6);
        REQUIRE(resHost.view()[4] == 9);
        REQUIRE(resHost.view()[5] == 12);
        REQUIRE(resHost.view()[6] == 15);
        REQUIRE(resHost.view()[7] == 17);
        REQUIRE(resHost.view()[8] == 19);
        REQUIRE(resHost.view()[9] == 21);
    }


    SECTION("Can unpack mtxValues " + execName)
    {
        auto newRowOffs = NeoN::la::stretchRowPtrs(rowOffs);
        auto res = NeoN::la::unpackMtxValues(mtxValues, rowOffs, newRowOffs);
        auto resHost = res.copyToHost();

        REQUIRE(resHost.view()[0] == 1.0);
        REQUIRE(resHost.view()[1] == 2.0);
        REQUIRE(resHost.view()[2] == 3.0);

        REQUIRE(resHost.view()[3] == 1.0);
        REQUIRE(resHost.view()[4] == 2.0);
        REQUIRE(resHost.view()[5] == 3.0);

        REQUIRE(resHost.view()[6] == 1.0);
        REQUIRE(resHost.view()[7] == 2.0);
        REQUIRE(resHost.view()[8] == 3.0);

        REQUIRE(resHost.view()[9] == 4.0);
        REQUIRE(resHost.view()[10] == 5.0);
        REQUIRE(resHost.view()[11] == 6.0);
    }

    SECTION("Can duplicateColIdx " + execName)
    {
        auto newRowOffs = NeoN::la::stretchRowPtrs(rowOffsS);
        auto res = NeoN::la::duplicateColIdx(colIdxS, newRowOffs, rowOffsS);
        auto resHost = res.copyToHost();

        REQUIRE(res.size() == 3 * colIdxS.size()); // 0

        // row 1
        REQUIRE(resHost.view()[0] == 0); // 0
        REQUIRE(resHost.view()[1] == 3); // 1

        // row 2
        REQUIRE(resHost.view()[2] == 1);
        REQUIRE(resHost.view()[3] == 4);

        // row 3
        REQUIRE(resHost.view()[4] == 2);
        REQUIRE(resHost.view()[5] == 5);

        // row 4
        REQUIRE(resHost.view()[6] == 0);
        REQUIRE(resHost.view()[7] == 3);
        REQUIRE(resHost.view()[8] == 6);

        // row 5
        REQUIRE(resHost.view()[9] == 1);
        REQUIRE(resHost.view()[10] == 4);
        REQUIRE(resHost.view()[11] == 7);

        // row 6
        REQUIRE(resHost.view()[12] == 2);
        REQUIRE(resHost.view()[13] == 5);
        REQUIRE(resHost.view()[14] == 8);

        // row 7
        REQUIRE(resHost.view()[15] == 3);
        REQUIRE(resHost.view()[16] == 6);

        // row 8
        REQUIRE(resHost.view()[17] == 4);
        REQUIRE(resHost.view()[18] == 7);

        // row 9
        REQUIRE(resHost.view()[19] == 5);
        REQUIRE(resHost.view()[20] == 8);
    }

    SECTION("Can duplicateColIdx " + execName)
    {
        auto newRowOffs = NeoN::la::stretchRowPtrs(rowOffs);
        auto res = NeoN::la::duplicateColIdx(colIdx, newRowOffs, rowOffs);
        auto resHost = res.copyToHost();

        REQUIRE(res.size() == 3 * colIdx.size()); // 0

        // row 1
        REQUIRE(resHost.view()[0] == 0); // 0
        REQUIRE(resHost.view()[1] == 3); // 1
        REQUIRE(resHost.view()[2] == 6); // 2

        // row 2
        REQUIRE(resHost.view()[3] == 1);
        REQUIRE(resHost.view()[4] == 4);
        REQUIRE(resHost.view()[5] == 7);

        // row 3
        REQUIRE(resHost.view()[6] == 2);
        REQUIRE(resHost.view()[7] == 5);
        REQUIRE(resHost.view()[8] == 8);

        // row 4
        REQUIRE(resHost.view()[9] == 0);
        REQUIRE(resHost.view()[10] == 3);
        REQUIRE(resHost.view()[11] == 6);

        // row 5
        REQUIRE(resHost.view()[12] == 1);
        REQUIRE(resHost.view()[13] == 4);
        REQUIRE(resHost.view()[14] == 7);

        // row 6
        REQUIRE(resHost.view()[15] == 2);
        REQUIRE(resHost.view()[16] == 5);
        REQUIRE(resHost.view()[17] == 8);

        // row 7
        REQUIRE(resHost.view()[18] == 0);
        REQUIRE(resHost.view()[19] == 3);
        REQUIRE(resHost.view()[20] == 6);

        // row 8
        REQUIRE(resHost.view()[21] == 1);
        REQUIRE(resHost.view()[22] == 4);
        REQUIRE(resHost.view()[23] == 7);

        // row 9
        REQUIRE(resHost.view()[24] == 2);
        REQUIRE(resHost.view()[25] == 5);
        REQUIRE(resHost.view()[26] == 8);
    }

    SECTION("Can compute residual on " + execName)
    {
        Vector<scalar> rhs(exec, 3, 2.0);
        Vector<scalar> x(exec, 3, 1.0);
        Vector<scalar> res(exec, 3, 0.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);

        NeoN::la::computeResidual(csrMatrix, rhs, x, res);

        auto resHost = res.copyToHost();

        REQUIRE(resHost.view()[0] == 4.0);
        REQUIRE(resHost.view()[1] == 13.0);
        REQUIRE(resHost.view()[2] == 22.0);
    }
}
