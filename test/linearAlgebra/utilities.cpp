// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
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
using NeoN::la::COOMatrix;

TEST_CASE("Utilities")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Dense matrix
    // [ 1 2 3 ]
    // [ 4 5 6 ]  - where in Vec3 variant each value is a Vec 3
    // [ 7 8 9 ]    e.g. 1 -> {1, 1, 1}
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
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs, {3, 3});

    // Sparse matrix variant of the above, i.e, not all rows contain
    // 3 entries
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

    SECTION("Can unpackhRowOffs " + execName)
    {
        auto res = NeoN::la::unpackRowOffs(rowOffs);
        auto rowOffsExp = std::vector<localIdx> {0, 3, 6, 9, 12, 15, 18, 21, 24, 27};
        REQUIRE_THAT(res, Equals(rowOffsExp, EqualInt()));
    }

    SECTION("Can unpackRowOffs2 " + execName)
    {
        auto res = NeoN::la::unpackRowOffs(rowOffsS);
        auto exp = std::vector<localIdx> {0, 2, 4, 6, 9, 12, 15, 17, 19, 21};
        REQUIRE_THAT(res, Equals(exp, EqualInt()));
    }

    SECTION("Can unpack mtxValues " + execName)
    {
        auto newRowOffs = NeoN::la::unpackRowOffs(rowOffs);
        auto res = NeoN::la::unpackMtxValues(mtxValues, rowOffs, newRowOffs);
        auto exp = std::vector<scalar> {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
                                        4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                        7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0};
        REQUIRE_THAT(res, Equals(exp));
    }

    SECTION("Can unpackColIdx sparse " + execName)
    {
        auto newRowOffs = NeoN::la::unpackRowOffs(rowOffsS);
        auto res = NeoN::la::unpackColIdx(colIdxS, newRowOffs, rowOffsS);

        REQUIRE(res.size() == 3 * colIdxS.size());

        auto colIdxExp = std::vector<localIdx> {
            0, 3,    // row 1
            1, 4,    // row 2
            2, 5,    // row 3
            0, 3, 6, // row 4
            1, 4, 7, // row 5
            2, 5, 8, // row 6
            3, 6,    // row 7
            4, 7,    // row 8
            5, 8     // row 9
        };
        REQUIRE_THAT(res, Equals(colIdxExp, EqualInt()));
    }

    SECTION("Can unpackColIdx dense " + execName)
    {
        auto newRowOffs = NeoN::la::unpackRowOffs(rowOffs);
        auto res = NeoN::la::unpackColIdx(colIdx, newRowOffs, rowOffs);

        REQUIRE(res.size() == 3 * colIdx.size());

        auto colIdxExp = std::vector<localIdx> {
            0, 3, 6, // row 1
            1, 4, 7, // row 2
            2, 5, 8, // row 3
            0, 3, 6, // row 4
            1, 4, 7, // row 5
            2, 5, 8, // row 6
            0, 3, 6, // row 7
            1, 4, 7, // row 8
            2, 5, 8  // row 9
        };
        REQUIRE_THAT(res, Equals(colIdxExp, EqualInt()));
    }

    // Residual of scalar matrix
    // [ 1 2 3 ]   [1]   [2]   [6]     [2]   [ 4]
    // [ 4 5 6 ] x [1] - [2] = [15]  - [2] = [13]
    // [ 7 8 9 ]   [1]   [2]   [24]    [2]   [22]
    SECTION("Can compute residual on " + execName)
    {
        Vector<scalar> rhs(exec, 3, 2.0);
        Vector<scalar> x(exec, 3, 1.0);
        Vector<scalar> res(exec, 3, 0.0);
        Vector<scalar> bValues(exec, {0.0, 0.0, 0.0});
        Vector<localIdx> bColIdx(exec, {0, 1, 2});
        Vector<localIdx> bRowOffs(exec, {0, 1, 2});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bColIdx, bRowOffs, {3, 1});
        LinearSystem<scalar, CSRMatrix<scalar, localIdx>> linearSystem(
            csrMatrix, rhs, bCooMatrix, rhs
        );

        NeoN::la::computeResidual(csrMatrix, rhs, x, res);

        auto residualExp = std::vector<scalar> {4.0, 13.0, 22.0};
        REQUIRE_THAT(res, Equals(residualExp, Approx(1e-15)));
    }

    SECTION("Can convert empty rowsToRowOffs  " + execName)
    {
        auto rowIdx = Vector<localIdx>(exec, {});
        auto rowOffs = NeoN::la::rowsToRowOffs(rowIdx);
        auto rowIdxExp = std::vector<localIdx> {0};
        REQUIRE_THAT(rowOffs, Equals(rowIdxExp, EqualInt()));
    }

    SECTION("Can convert non empty rowsToRowOffs  " + execName)
    {
        auto rowIdx = Vector<localIdx>(exec, {0, 1, 2});
        auto rowOffs = NeoN::la::rowsToRowOffs(rowIdx);
        auto rowIdxExp = std::vector<localIdx> {0, 1, 2, 3};
        REQUIRE_THAT(rowOffs, Equals(rowIdxExp, EqualInt()));
    }

    SECTION("Can convert non empty with duplicates rowsToRowOffs  " + execName)
    {
        auto rowIdx = Vector<localIdx>(exec, {0, 0, 2, 2, 2});
        auto rowOffs = NeoN::la::rowsToRowOffs(rowIdx);
        auto rowIdxExp = std::vector<localIdx> {0, 2, 2, 5};
        REQUIRE_THAT(rowOffs, Equals(rowIdxExp, EqualInt()));
    }
}
