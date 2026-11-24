// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include "NeoN/NeoN.hpp"


#if NF_WITH_GINKGO

using NeoN::Executor;
using NeoN::Dictionary;
using NeoN::scalar;
using NeoN::Vec3;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CsrSparsityPattern;
using NeoN::la::CooSparsityPattern;
using NeoN::la::CSRMatrix;
using NeoN::la::COOMatrix;
using NeoN::la::Solver;
using NeoN::la::Dimensions;

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

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
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
    SECTION("Nested solver::Ir keeps its inner solver")
    {
        // Ir names its inner solver 'solver'; inside a sub-dictionary that is not the backend key.
        NeoN::Dictionary dict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"preconditioner",
              NeoN::Dictionary {
                  {{"type", "solver::Ir"},
                   {"solver", NeoN::Dictionary {{{"type", "preconditioner::Jacobi"}}}}}
              }}}
        };

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"type", gko::config::pnode {"solver::Cg"}},
             {"preconditioner",
              gko::config::pnode(
                  {{"type", gko::config::pnode {"solver::Ir"}},
                   {"solver",
                    gko::config::pnode({{"type", gko::config::pnode {"preconditioner::Jacobi"}}})}}
              )}}
        );
        CHECK(node == expected);
    }
    SECTION("Top-level solver::Ir takes its inner solver from 'preconditioner'")
    {
        NeoN::Dictionary dict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", std::string {"solver::Ir"}},
             {"preconditioner", NeoN::Dictionary {{{"type", "preconditioner::Jacobi"}}}}}
        };

        auto node = NeoN::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"type", gko::config::pnode {"solver::Ir"}},
             {"solver",
              gko::config::pnode({{"type", gko::config::pnode {"preconditioner::Jacobi"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoN::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoN::la::ginkgo::parse(dict), NeoN::NeoNException);
    }
}

TEST_CASE("readFlag - Ginkgo")
{
    using NeoN::la::ginkgo::readFlag;

    // A boolean read from a dictionary file arrives as a word, e.g. reuseSetup false;
    NeoN::Dictionary dict {
        {{"bool", false}, {"int", 1}, {"word", std::string("false")}, {"yes", std::string("yes")}}
    };

    CHECK(readFlag(dict, "bool", true) == false);
    CHECK(readFlag(dict, "int", false) == true);
    CHECK(readFlag(dict, "word", true) == false);
    CHECK(readFlag(dict, "yes", false) == true);
    CHECK(readFlag(dict, "missing", true) == true);
    CHECK(readFlag(dict, "missing", false) == false);
}

TEST_CASE("gkoVecView - Ginkgo")
{
    NeoN::Executor exec = NeoN::SerialExecutor {};
    auto gkoExec = NeoN::la::ginkgo::getGkoExecutor(exec);

    SECTION("scalar mutable: 1-column non-owning Dense")
    {
        localIdx n = 4;
        Vector<scalar> v(exec, {1.0, 2.0, 3.0, 4.0});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {1});
        CHECK(dense->get_stride() == gko::size_type {1});
        CHECK(dense->get_values() == v.data());
    }

    SECTION("scalar const: 1-column non-owning Dense")
    {
        localIdx n = 4;
        const Vector<scalar> v(exec, {1.0, 2.0, 3.0, 4.0});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {1});
        CHECK(dense->get_stride() == gko::size_type {1});
        CHECK(dense->get_const_values() == v.data());
    }

    SECTION("Vec3 mutable: 3-column non-owning Dense")
    {
        localIdx n = 3;
        Vector<Vec3> v(exec, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {3});
        CHECK(dense->get_stride() == gko::size_type {3});
        CHECK(dense->get_values() == reinterpret_cast<scalar*>(v.data()));
    }

    SECTION("Vec3 const: 3-column non-owning Dense")
    {
        localIdx n = 3;
        const Vector<Vec3> v(exec, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        auto dense = NeoN::la::ginkgo::gkoVecView(gkoExec, v.data(), n);

        CHECK(dense->get_size()[0] == static_cast<gko::size_type>(n));
        CHECK(dense->get_size()[1] == gko::size_type {3});
        CHECK(dense->get_stride() == gko::size_type {3});
        CHECK(dense->get_const_values() == reinterpret_cast<const scalar*>(v.data()));
    }
}

TEST_CASE("MatrixConversion - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto values = Vector<scalar>(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto rowIdx = Vector<localIdx>(exec, {0, 0, 1, 1, 1, 2, 2, 2, 3, 3});
    auto colIdx = Vector<localIdx>(exec, {0, 1, 0, 1, 2, 1, 2, 3, 2, 3});
    auto rowPtr = Vector<localIdx>(exec, {0, 2, 5, 8, 10});

    SECTION("CSRMatrix " + execName)
    {
        auto csrMatrix = CSRMatrix<scalar, localIdx>(values, colIdx, rowPtr, {4, 4});
        auto gkoCsrMtx = NeoN::la::ginkgo::createGkoMtx(csrMatrix);
    }

    SECTION("COOMatrix " + execName)
    {
        auto cooMatrix = COOMatrix<scalar, localIdx>(values, colIdx, rowIdx, {4, 4});
        auto gkoCooMtx = NeoN::la::ginkgo::createGkoMtx(cooMatrix);
    }
}

TEST_CASE("MatrixAssembly - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    Vector<localIdx> colIdx(exec, {0, 1, 0, 1, 2, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 2, 5, 7});
    Vector<localIdx> bColIdx(exec, {});
    Vector<localIdx> bRowOffs(exec, {});

    const auto nRows = static_cast<localIdx>(rowOffs.size()) - 1;
    auto sparsity = std::make_shared<CsrSparsityPattern<localIdx>>(
        std::move(colIdx), std::move(rowOffs), Dimensions {nRows, nRows}
    );
    auto bSparsity = std::make_shared<CooSparsityPattern<localIdx>>(
        std::move(bColIdx), std::move(bRowOffs), Dimensions {0, 0}
    );

    SECTION("Solve linear system scalar " + execName)
    {
        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});

        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});

        auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = NeoN::la::Solver(exec, solverDict);

        // Solve system
        auto solverStats = solver.solve(linearSystem, x);
        auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

        auto hostX = x.copyToHost();
        auto hostXS = hostX.view();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
        REQUIRE(numIter == 3);
        REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
        REQUIRE(finalResNorm < 1.0e-04);
    }

    SECTION("configFile with the L1 criterion reports the scaled residual " + execName)
    {
        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});
        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});
        auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);
        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        // The criterion is named in the config, so it runs inside the solver; the stats must
        // carry its scaled L1 residual, not the unscaled L2 norm of the generic solve path.
        const auto configFile =
            (std::filesystem::temp_directory_path() / "neon_test_l1_config.json").string();
        std::ofstream(configFile) << R"({"type": "solver::Cg", "criteria": [)"
                                  << R"({"type": "Iteration", "max_iters": 20}, )"
                                  << R"("neon::l1ScaledResidual"]})";
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"configFile", configFile},
             {"l1ScaledResidual", true},
             {"tolerance", 1e-12},
             {"relTol", 0.0}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);
        auto solverStats = solver.solve(linearSystem, x);
        std::filesystem::remove(configFile);
        auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

        // From x = 0 the scaled residual is |b| / |b| = 1; the unscaled L2 norm would be 3.74.
        REQUIRE(initResNorm == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(finalResNorm < 1e-10);
        REQUIRE(numIter > 0);
        auto hostX = x.copyToHost();
        REQUIRE(hostX.view()[0] == Catch::Approx(1.24489796).margin(1e-8));
    }

    SECTION("configFile with the L1 criterion reports another criterion's stop " + execName)
    {
        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});
        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});
        auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);
        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        // The Iteration cap stops after 2 iterations, between the L1 criterion's checks (every
        // 3rd iteration), so its report still describes iteration 0.
        const auto configFile =
            (std::filesystem::temp_directory_path() / "neon_test_l1_config_cap.json").string();
        std::ofstream(configFile) << R"({"type": "solver::Cg", "criteria": [)"
                                  << R"({"type": "Iteration", "max_iters": 2}, )"
                                  << R"("neon::l1ScaledResidual"]})";
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"configFile", configFile},
             {"l1ScaledResidual", true},
             {"checkFrequency", 3},
             {"tolerance", 1e-12},
             {"relTol", 0.0}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);
        auto solverStats = solver.solve(linearSystem, x);
        std::filesystem::remove(configFile);
        auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

        // sum|b - A x| / normFactor, with normFactor = sum|b| = 6 at x = 0.
        auto hostX = x.copyToHost();
        auto xs = hostX.view();
        const scalar l1Residual = std::abs(1.0 - (xs[0] - 0.1 * xs[1]))
                                + std::abs(2.0 - (-0.1 * xs[0] + xs[1] - 0.1 * xs[2]))
                                + std::abs(3.0 - (-0.1 * xs[1] + xs[2]));
        REQUIRE(numIter == 2);
        REQUIRE(initResNorm == Catch::Approx(1.0).margin(1e-12));
        REQUIRE(finalResNorm == Catch::Approx(l1Residual / 6.0).margin(1e-12));
        REQUIRE(finalResNorm < 0.1);
    }

    SECTION("Solve linear system scalar with solver::Ir " + execName)
    {
        Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
        Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});

        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});

        auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        Vector<scalar> x(exec, {0.0, 0.0, 0.0});

        // Richardson iteration with a Jacobi inner solver, given under 'preconditioner' because
        // 'solver' selects the Ginkgo backend. Unlike the Krylov solvers, Ginkgo's Ir only takes
        // typed criteria, not the short form {iteration: ..., relative_residual_norm: ...}.
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Ir"},
             {"preconditioner", Dictionary {{{"type", "preconditioner::Jacobi"}}}},
             {"criteria", Dictionary {{{"type", "Iteration"}, {"max_iters", 100}}}}}
        };

        auto solver = NeoN::la::Solver(exec, solverDict);
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.view();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
    }

    // The short form {iteration: .., <norm>: ..} is what NeoFOAM's fvSolution mapping writes;
    // Ginkgo's Ir itself only parses typed criteria, so parse() has to translate it.
    for (const bool l1 : {false, true})
    {
        SECTION(
            std::string("solver::Ir with short-form criteria") + (l1 ? ", l1ScaledResidual " : " ")
            + execName
        )
        {
            Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
            CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
            Vector<scalar> rhs(exec, {1.0, 2.0, 3.0});
            Vector<scalar> bValues(exec, {});
            COOMatrix<scalar, localIdx> bCooMatrix(bValues, bSparsity);
            Vector<scalar> bRhs(exec, {});
            auto linearSystem = LinearSystem<scalar>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);
            Vector<scalar> x(exec, {0.0, 0.0, 0.0});

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Ir"},
                 {"preconditioner", Dictionary {{{"type", "preconditioner::Jacobi"}}}},
                 {"criteria", Dictionary {{{"iteration", 100}, {"absolute_residual_norm", 1e-12}}}}}
            };
            if (l1)
            {
                solverDict.insert("l1ScaledResidual", true);
            }

            auto solver = NeoN::la::Solver(exec, solverDict);
            auto solverStats = solver.solve(linearSystem, x);

            auto hostX = x.copyToHost();
            auto hostXS = hostX.view();
            REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
            REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
            REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
            REQUIRE(solverStats.entries[0].numIter < 100);
        }
    }

    SECTION("Solve linear system vector " + execName)
    {
        Vector<Vec3> values(
            exec,
            {{1.0, 1.0, 1.0},
             {-0.1, -0.1, -0.1},
             {-0.1, -0.1, -0.1},
             {1.0, 1.0, 1.0},
             {-0.1, -0.1, -0.1},
             {-0.1, -0.1, -0.1},
             {1.0, 1.0, 1.0}}
        );

        CSRMatrix<Vec3, localIdx> csrMatrix(values, sparsity);
        Vector<Vec3> bValues(exec, {});
        COOMatrix<Vec3, localIdx> bCooMatrix(bValues, bSparsity);
        Vector<Vec3> bRhs(exec, {});

        Vector<Vec3> rhs(exec, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}});
        Vector<Vec3> x(exec, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});

        auto linearSystem = LinearSystem<Vec3>(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        SECTION("Segregated" + execName)
        {

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"coupled", false},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            for (auto entry : solverStats.entries)
            {
                auto [numIter, initResNorm, finalResNorm, solveTime] = entry;
                auto hostX = x.copyToHost();
                auto hostXS = hostX.view();
                REQUIRE((hostXS[0][0]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][0]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][0]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][1]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][1]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][1]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][2]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][2]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][2]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE(numIter == 3);
                REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
                REQUIRE(finalResNorm < 1.0e-04);
            }
        }
        SECTION("Coupled" + execName)
        {

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"coupled", true},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            for (auto entry : solverStats.entries)
            {
                auto [numIter, initResNorm, finalResNorm, solveTime] = entry;
                auto hostX = x.copyToHost();
                auto hostXS = hostX.view();
                REQUIRE((hostXS[0][0]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][0]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][0]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][1]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][1]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][1]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE((hostXS[0][2]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][2]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][2]) == Catch::Approx(3.24489796).margin(1e-8));

                REQUIRE(numIter == 3);
                REQUIRE(initResNorm == Catch::Approx(6.4807406984).margin(1e-8));
                REQUIRE(finalResNorm < 1.0e-04);
            }
        }

        SECTION("Solve linear system wo boundary scalar with multiple rhs " + execName)
        {
            Vector<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
            CSRMatrix<scalar, localIdx> csrMatrix(values, sparsity);
            Vector<Vec3> rhs(exec, {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}});

            Vector<scalar> bValues(exec, {});
            COOMatrix<scalar, localIdx> bCsrMatrix(bValues, bSparsity);
            Vector<Vec3> bRhs(exec, {});

            auto linearSystem = LinearSystem<
                scalar,
                NeoN::Vec3,
                NeoN::la::CSRMatrix<scalar, NeoN::localIdx>,
                NeoN::la::COOMatrix<scalar, NeoN::localIdx>>(csrMatrix, rhs, bCsrMatrix, bRhs);

            Vector<Vec3> x(exec, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});

            Dictionary solverDict {
                {{"solver", std::string {"Ginkgo"}},
                 {"type", "solver::Cg"},
                 {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
            };

            // Create solver
            auto solver = NeoN::la::Solver(exec, solverDict);

            // Solve system
            auto solverStats = solver.solve(linearSystem, x);
            auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];

            auto hostX = x.copyToHost();
            auto hostXS = hostX.view();
            for (int c = 0; c < 3; ++c)
            {
                REQUIRE((hostXS[0][c]) == Catch::Approx(1.24489796).margin(1e-8));
                REQUIRE((hostXS[1][c]) == Catch::Approx(2.44897959).margin(1e-8));
                REQUIRE((hostXS[2][c]) == Catch::Approx(3.24489796).margin(1e-8));
            }
            REQUIRE(numIter == 3);
            REQUIRE(initResNorm == Catch::Approx(3.741657386).margin(1e-8));
            REQUIRE(finalResNorm < 1.0e-04);
        }
    }
}

// Exercises the implicit transform-BC solver path: a scalar matrix with a Vec3 RHS plus a
// per-component diagonal correction (diagCmpt). The three components are solved segregated, with
// each column's correction temporarily subtracted from the shared diagonal and then restored.
// Using a purely diagonal matrix gives an analytic answer x_c = b_c / (D - diagCmpt_c).
TEST_CASE("Implicit transform diagonal correction solve - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 4;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);
    auto ls = NeoN::la::createEmptyLinearSystem<scalar, Vec3>(mesh);

    // diagonal D·I (off-diagonals left at zero)
    const scalar D = 10.0;
    {
        auto values = ls.matrix().values().view();
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().rowOffs().view());
        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx c) { values[ma.diagIdx(c)] = D; },
            "setDiag"
        );
    }

    // per-component diagonal correction (subtracted by the solver) and a uniform RHS
    const Vec3 dc(1.0, 2.0, 3.0);
    NeoN::fill(ls.ensureDiagCmpt(), dc);
    const scalar bVal = 6.0;
    NeoN::fill(ls.rhs(), Vec3(bVal, bVal, bVal));

    // analytic per-component solution: x_c = b_c / (D - diagCmpt_c)
    auto requireAnalytic = [&](const Vector<Vec3>& sol)
    {
        auto host = sol.copyToHost();
        auto v = host.view();
        for (localIdx i = 0; i < nCells; ++i)
        {
            REQUIRE(v[i][0] == Catch::Approx(bVal / (D - dc[0])).margin(1e-8));
            REQUIRE(v[i][1] == Catch::Approx(bVal / (D - dc[1])).margin(1e-8));
            REQUIRE(v[i][2] == Catch::Approx(bVal / (D - dc[2])).margin(1e-8));
        }
    };

    SECTION("standard stopping criterion " + execName)
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 20}, {"relative_residual_norm", 1e-10}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);

        Vector<Vec3> x(exec, nCells, Vec3(0.0, 0.0, 0.0));
        auto stats = solver.solve(ls, x);
        REQUIRE(stats.entries.size() == 3); // three segregated component solves
        requireAnalytic(x);

        // Re-solving must give the same answer: only holds if the shared diagonal was restored
        // after each column (otherwise it would be doubly corrected, D - 2·dc).
        Vector<Vec3> x2(exec, nCells, Vec3(0.0, 0.0, 0.0));
        solver.solve(ls, x2);
        requireAnalytic(x2);
    }

    SECTION("l1ScaledResidual stopping criterion " + execName)
    {
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"l1ScaledResidual", true},
             {"criteria", Dictionary {{{"iteration", 20}, {"absolute_residual_norm", 1e-10}}}}}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);

        Vector<Vec3> x(exec, nCells, Vec3(0.0, 0.0, 0.0));
        auto stats = solver.solve(ls, x);
        REQUIRE(stats.entries.size() == 3);
        requireAnalytic(x);
    }
}
#endif
