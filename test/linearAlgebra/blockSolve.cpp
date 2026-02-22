// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockSolve.hpp"

TEST_CASE("BlockSolve")
{
    using namespace NeoN;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Two coupled scalars on single cell " + execName)
    {
        //   [4  1] [a]   [11]     →  a = 2, b = 3
        //   [1  3] [b] = [11]
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );
        // Interleaved column-major at pos 0: (0,0)=4, (1,0)=1, (0,1)=1, (1,1)=3
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp, vals);

        BlockVector rhsVec(exec, 2, 1, 0.0);
        rhsVec.copyBlockFrom(0, Vector<scalar>(exec, std::vector<scalar> {11.0}));
        rhsVec.copyBlockFrom(1, Vector<scalar>(exec, std::vector<scalar> {11.0}));

        BlockVector solution(exec, 2, 1, 0.0);

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 10}, {"relative_residual_norm", 1e-10}}}}}
        };

        auto stats = la::solve(bm, rhsVec, solution, solverDict);

        Vector<scalar> solA(exec, 1, 0.0);
        Vector<scalar> solB(exec, 1, 0.0);
        solution.copyBlockTo(0, solA);
        solution.copyBlockTo(1, solB);

        auto hostA = solA.copyToHost();
        auto hostB = solB.copyToHost();
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
        // Interleaved column-major at pos 0: (0,0)=4, (1,0)=0, (0,1)=0, (1,1)=3
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 0.0, 0.0, 3.0});
        BlockMatrix bm(exec, 2, sp, vals);

        BlockVector rhsVec(exec, 2, 1, 0.0);
        rhsVec.copyBlockFrom(0, Vector<scalar>(exec, std::vector<scalar> {8.0}));
        rhsVec.copyBlockFrom(1, Vector<scalar>(exec, std::vector<scalar> {9.0}));

        BlockVector solution(exec, 2, 1, 0.0);
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 10}, {"relative_residual_norm", 1e-10}}}}}
        };

        la::solve(bm, rhsVec, solution, solverDict);

        Vector<scalar> solA(exec, 1, 0.0);
        Vector<scalar> solB(exec, 1, 0.0);
        solution.copyBlockTo(0, solA);
        solution.copyBlockTo(1, solB);

        auto hostA = solA.copyToHost();
        auto hostB = solB.copyToHost();
        REQUIRE(hostA.view()[0] == Catch::Approx(2.0).margin(1e-8));
        REQUIRE(hostB.view()[0] == Catch::Approx(3.0).margin(1e-8));
    }

    SECTION("Two coupled fields on 3-cell mesh " + execName)
    {
        auto sp3 = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 2, 5, 7})
        );

        // Interleaved layout: 7 coupling matrices, each 2x2 column-major [a00, a10, a01, a11]
        // Diagonal couplings: (0,0)=2, (1,0)=0.1, (0,1)=0.1, (1,1)=2
        // Off-diagonal couplings: (0,0)=-1, (1,0)=0, (0,1)=0, (1,1)=-1
        std::vector<scalar> vals = {
            2,  0.1, 0.1, 2,  // pos 0 (0->0)
            -1, 0,   0,   -1, // pos 1 (0->1)
            -1, 0,   0,   -1, // pos 2 (1->0)
            2,  0.1, 0.1, 2,  // pos 3 (1->1)
            -1, 0,   0,   -1, // pos 4 (1->2)
            -1, 0,   0,   -1, // pos 5 (2->1)
            2,  0.1, 0.1, 2   // pos 6 (2->2)
        };
        BlockMatrix bm(exec, 2, sp3, Vector<scalar>(exec, vals));

        // RHS: A * [1,1,1,1,1,1] = [1.1, 0.1, 1.1, 1.1, 0.1, 1.1]
        BlockVector rhsVec(exec, 2, 3, 0.0);
        rhsVec.copyBlockFrom(0, Vector<scalar>(exec, std::vector<scalar> {1.1, 0.1, 1.1}));
        rhsVec.copyBlockFrom(1, Vector<scalar>(exec, std::vector<scalar> {1.1, 0.1, 1.1}));

        BlockVector solution(exec, 2, 3, 0.0);
        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 100}, {"relative_residual_norm", 1e-10}}}}}
        };

        la::solve(bm, rhsVec, solution, solverDict);

        Vector<scalar> solA(exec, 3, 0.0);
        Vector<scalar> solB(exec, 3, 0.0);
        solution.copyBlockTo(0, solA);
        solution.copyBlockTo(1, solB);

        auto hostA = solA.copyToHost();
        auto hostB = solB.copyToHost();
        for (localIdx i = 0; i < 3; ++i)
        {
            REQUIRE(hostA.view()[i] == Catch::Approx(1.0).margin(1e-6));
            REQUIRE(hostB.view()[i] == Catch::Approx(1.0).margin(1e-6));
        }
    }
}
