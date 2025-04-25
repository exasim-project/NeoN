// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::spmv;

TEST_CASE("Utilities")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // [ 1 2 3 ]   [1]   [2]   [6]     [2]
    // [ 4 5 6 ] x [1] - [2] = [15]  - [2]
    // [ 7 8 9 ]   [1]   [2]   [24]    [2]
    Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Vector<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs);

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
