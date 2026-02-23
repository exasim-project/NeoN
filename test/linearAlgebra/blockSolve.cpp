// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockDsl.hpp"
#include "NeoN/linearAlgebra/blockLinearSystem.hpp"

TEST_CASE("BlockSolve")
{
    using namespace NeoN;
    using namespace NeoN::la;
    using namespace NeoN::bdsl;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Two coupled scalars on single cell " + execName)
    {
        //   [4  1] [a]   [11]     →  a = 2, b = 3
        //   [1  3] [b] = [11]
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );

        Vector<scalar> a(exec, 1, 0.0);
        Vector<scalar> b(exec, 1, 0.0);

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 10}, {"relative_residual_norm", 1e-10}}}}}
        };

        BlockLinearSystem system(exec, {"a", "b"}, {&a, &b}, sp, solverDict);

        auto [aExpr, bExpr] = system.expressions<2>();
        aExpr = imp::source(4.0, a, "a") + imp::source(1.0, b, "b");
        bExpr = imp::source(1.0, a, "a") + imp::source(3.0, b, "b");

        system.setRhs(0, Vector<scalar>(exec, std::vector<scalar> {11.0}));
        system.setRhs(1, Vector<scalar>(exec, std::vector<scalar> {11.0}));

        system.assemble();
        system.solve();

        auto hostA = a.copyToHost();
        auto hostB = b.copyToHost();
        REQUIRE(hostA.view()[0] == Catch::Approx(2.0).margin(1e-8));
        REQUIRE(hostB.view()[0] == Catch::Approx(3.0).margin(1e-8));
    }

    SECTION("Diagonal-only blocks (decoupled) " + execName)
    {
        // [4  0] [a]   [8]     →  a = 2, b = 3
        // [0  3] [b] = [9]
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );

        Vector<scalar> a(exec, 1, 0.0);
        Vector<scalar> b(exec, 1, 0.0);

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 10}, {"relative_residual_norm", 1e-10}}}}}
        };

        BlockLinearSystem system(exec, {"a", "b"}, {&a, &b}, sp, solverDict);

        system.expression(0) = imp::source(4.0, a, "a");
        system.expression(1) = imp::source(3.0, b, "b");

        system.setRhs(0, Vector<scalar>(exec, std::vector<scalar> {8.0}));
        system.setRhs(1, Vector<scalar>(exec, std::vector<scalar> {9.0}));

        system.assemble();
        system.solve();

        auto hostA = a.copyToHost();
        auto hostB = b.copyToHost();
        REQUIRE(hostA.view()[0] == Catch::Approx(2.0).margin(1e-8));
        REQUIRE(hostB.view()[0] == Catch::Approx(3.0).margin(1e-8));
    }
}
