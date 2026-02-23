// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockDsl.hpp"
#include "NeoN/linearAlgebra/blockLinearSystem.hpp"

// Cycle 1: BlockSourceTerm — construction and getters
TEST_CASE("BlockSourceTerm")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;

    SECTION("construction and getters")
    {
        BlockSourceTerm st(4.0, "velocity");

        REQUIRE(st.getFieldName() == "velocity");
        REQUIRE(st.getName() == "BlockSourceTerm");
        REQUIRE(st.coefficient() == 4.0);
    }
}

// Cycle 2: BlockSourceTerm::implicitOperation — writes diagonal coupling
TEST_CASE("BlockSourceTerm implicitOperation")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("writes diagonal coupling on 1-cell " + execName)
    {
        // 1-cell sparsity: colIdxs={0}, rowOffs={0,1}
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );

        // 2x2 block matrix on 1 cell
        BlockMatrix bm(exec, 2, sp);
        auto bmView = bm.view();
        auto spView = sp->view();

        BlockSourceTerm st(4.0, "a");
        st.implicitOperation(bmView, spView, 0, 0, 1, exec);

        auto hostVals = bm.values().copyToHost();
        // Block at position 0: 2x2 column-major = [val(0,0), val(1,0), val(0,1), val(1,1)]
        REQUIRE(hostVals.view()[0] == Catch::Approx(4.0)); // (0,0)
        REQUIRE(hostVals.view()[1] == Catch::Approx(0.0)); // (1,0)
        REQUIRE(hostVals.view()[2] == Catch::Approx(0.0)); // (0,1)
        REQUIRE(hostVals.view()[3] == Catch::Approx(0.0)); // (1,1)
    }
}

// Cycle 3: SpatialOperator type-erasure
TEST_CASE("bdsl::SpatialOperator type-erasure")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;

    SECTION("wraps BlockSourceTerm")
    {
        SpatialOperator<scalar> op(BlockSourceTerm(4.0, "velocity"));

        REQUIRE(op.getFieldName() == "velocity");
        REQUIRE(op.getName() == "BlockSourceTerm");
    }

    SECTION("copy construction")
    {
        SpatialOperator<scalar> op1(BlockSourceTerm(4.0, "velocity"));
        SpatialOperator<scalar> op2(op1);

        REQUIRE(op2.getFieldName() == "velocity");
        REQUIRE(op2.getName() == "BlockSourceTerm");
    }
}

// Cycle 4: bdsl::imp::source free function
TEST_CASE("bdsl::imp::source")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;

    auto exec = SerialExecutor {};
    Vector<scalar> a(exec, 1, 0.0);

    auto op = imp::source(4.0, a, "a");

    REQUIRE(op.getFieldName() == "a");
    REQUIRE(op.getName() == "BlockSourceTerm");
}

// Cycle 5: operator+ composition
TEST_CASE("bdsl operator+ composition")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;

    auto exec = SerialExecutor {};
    Vector<scalar> a(exec, 1, 0.0);
    Vector<scalar> b(exec, 1, 0.0);

    SECTION("two operators")
    {
        auto ops = imp::source(4.0, a, "a") + imp::source(1.0, b, "b");

        REQUIRE(ops.size() == 2);
        REQUIRE(ops[0].getFieldName() == "a");
        REQUIRE(ops[1].getFieldName() == "b");
    }

    SECTION("three operators")
    {
        Vector<scalar> c(exec, 1, 0.0);
        auto ops = imp::source(4.0, a, "a") + imp::source(1.0, b, "b") + imp::source(2.0, c, "c");

        REQUIRE(ops.size() == 3);
        REQUIRE(ops[0].getFieldName() == "a");
        REQUIRE(ops[1].getFieldName() == "b");
        REQUIRE(ops[2].getFieldName() == "c");
    }
}

// Cycle 6: BlockExpression — stores operators, routes by name
TEST_CASE("bdsl::BlockExpression")
{
    using namespace NeoN;
    using namespace NeoN::bdsl;

    auto exec = SerialExecutor {};
    Vector<scalar> a(exec, 1, 0.0);
    Vector<scalar> b(exec, 1, 0.0);

    SECTION("construction and field column lookup")
    {
        BlockExpression<scalar> expr(0, {"a", "b"});

        REQUIRE(expr.fieldColumn("a") == 0);
        REQUIRE(expr.fieldColumn("b") == 1);
        REQUIRE(expr.fieldColumn("c") == -1);
    }

    SECTION("assign single operator")
    {
        BlockExpression<scalar> expr(0, {"a", "b"});
        expr = imp::source(4.0, a, "a");

        REQUIRE(expr.operators().size() == 1);
        REQUIRE(expr.operators()[0].getFieldName() == "a");
    }

    SECTION("assign vector of operators")
    {
        BlockExpression<scalar> expr(0, {"a", "b"});
        expr = imp::source(4.0, a, "a") + imp::source(1.0, b, "b");

        REQUIRE(expr.operators().size() == 2);
        REQUIRE(expr.operators()[0].getFieldName() == "a");
        REQUIRE(expr.operators()[1].getFieldName() == "b");
    }
}

// Cycle 7: BlockLinearSystem integration — new constructor + bdsl assemble + solve
TEST_CASE("BlockLinearSystem with bdsl")
{
    using namespace NeoN;
    using namespace NeoN::la;
    using namespace NeoN::bdsl;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("2x2 coupled system on single cell " + execName)
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

    SECTION("Diagonal-only (decoupled) " + execName)
    {
        //   [4  0] [a]   [8]     →  a = 2, b = 3
        //   [0  3] [b] = [9]
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

    SECTION("3-cell mesh with diagonal source terms " + execName)
    {
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 2, 3})
        );

        Vector<scalar> a(exec, 3, 0.0);
        Vector<scalar> b(exec, 3, 0.0);

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 100}, {"relative_residual_norm", 1e-10}}}}}
        };

        BlockLinearSystem system(exec, {"a", "b"}, {&a, &b}, sp, solverDict);

        auto [aExpr, bExpr] = system.expressions<2>();
        aExpr = imp::source(4.0, a, "a") + imp::source(1.0, b, "b");
        bExpr = imp::source(1.0, a, "a") + imp::source(3.0, b, "b");

        system.setRhs(0, Vector<scalar>(exec, std::vector<scalar> {11.0, 11.0, 11.0}));
        system.setRhs(1, Vector<scalar>(exec, std::vector<scalar> {11.0, 11.0, 11.0}));

        system.assemble();
        system.solve();

        auto hostA = a.copyToHost();
        auto hostB = b.copyToHost();
        for (localIdx i = 0; i < 3; ++i)
        {
            REQUIRE(hostA.view()[i] == Catch::Approx(2.0).margin(1e-8));
            REQUIRE(hostB.view()[i] == Catch::Approx(3.0).margin(1e-8));
        }
    }
}
