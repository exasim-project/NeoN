// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


#if NF_WITH_GINKGO

using NeoN::Executor;
using NeoN::Dictionary;
using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::Solver;

TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoN::Dictionary dict {{{"key", std::string("value")}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Const Char *")
    {
        NeoN::Dictionary dict {{{"key", "value"}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Int")
    {
        NeoN::Dictionary dict {{{"key", 10}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK(node == expected);
    }
    SECTION("Double")
    {
        NeoN::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Float")
    {
        NeoN::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0f}}});
        CHECK(node == expected);
    }
    SECTION("Dict")
    {
        NeoN::Dictionary dict;
        dict.insert("key", NeoN::Dictionary {{"key", "value"}});

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoN::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoN::la::ginkgo::parse(dict), NeoN::NeoNException);
    }
}

TEST_CASE("MatrixAssembly - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("Solve linear system " + execName)
    {

        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        Vector<localIdx> colIdx(exec, {0, 1, 0, 1, 2, 1, 2});
        Vector<localIdx> rowOffs(exec, {0, 2, 5, 7});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs);

        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = NeoN::la::Solver(exec, solverDict);

        // Solve system
        auto [numIter, initResNorm, finalResNorm, solveTime] = solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.view();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
        REQUIRE(numIter == 3);
        REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
        REQUIRE(finalResNorm < 1.0e-04);
    }
}
#endif
