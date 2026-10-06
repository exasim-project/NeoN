// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#if NF_WITH_GINKGO

#include <filesystem>
#include <fstream>
#include <string>

using NeoN::Dictionary;
using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vector;
using NeoN::la::COOMatrix;
using NeoN::la::CooSparsityPattern;
using NeoN::la::CSRMatrix;
using NeoN::la::CsrSparsityPattern;
using NeoN::la::Dimensions;
using NeoN::la::LinearSystem;

using MergedPgm = NeoN::la::ginkgo::MergedPgm<double, gko::int32>;
using Csr = gko::matrix::Csr<double, gko::int32>;
using Dense = gko::matrix::Dense<double>;

namespace
{

/** @brief 1D Laplacian tridiag(-1, d, -1), large enough for several Pgm coarsening steps. */
std::shared_ptr<Csr>
laplacian1D(std::shared_ptr<const gko::Executor> exec, gko::size_type n, double d = 2.0)
{
    gko::matrix_data<double, gko::int32> data {gko::dim<2> {n, n}};
    for (gko::size_type i = 0; i < n; ++i)
    {
        const auto row = static_cast<gko::int32>(i);
        if (i > 0) data.nonzeros.emplace_back(row, row - 1, -1.0);
        data.nonzeros.emplace_back(row, row, d);
        if (i + 1 < n) data.nonzeros.emplace_back(row, row + 1, -1.0);
    }
    auto mtx = gko::share(Csr::create(exec));
    mtx->read(data);
    return mtx;
}

std::vector<double> values(const gko::LinOp* op)
{
    gko::matrix_data<double, gko::int32> data;
    gko::as<Csr>(op)->write(data);
    std::vector<double> out;
    for (const auto& nz : data.nonzeros)
        out.push_back(nz.value);
    return out;
}

/** @brief NeoN system for tridiag(-1, d, -1) with rhs = A * ones (exact solution all ones). */
LinearSystem<scalar> laplacianSystem(const NeoN::Executor& exec, localIdx n, scalar d)
{
    std::vector<localIdx> cols;
    std::vector<localIdx> rows {0};
    std::vector<scalar> vals;
    std::vector<scalar> rhs;
    for (localIdx i = 0; i < n; ++i)
    {
        scalar rowSum = d;
        if (i > 0)
        {
            cols.push_back(i - 1);
            vals.push_back(-1.0);
            rowSum -= 1.0;
        }
        cols.push_back(i);
        vals.push_back(d);
        if (i < n - 1)
        {
            cols.push_back(i + 1);
            vals.push_back(-1.0);
            rowSum -= 1.0;
        }
        rows.push_back(static_cast<localIdx>(cols.size()));
        rhs.push_back(rowSum);
    }
    auto sparsity = std::make_shared<CsrSparsityPattern<localIdx>>(
        Vector<localIdx>(exec, cols), Vector<localIdx>(exec, rows), Dimensions {n, n}
    );
    auto bSparsity = std::make_shared<CooSparsityPattern<localIdx>>(
        Vector<localIdx>(exec, {}), Vector<localIdx>(exec, {}), Dimensions {0, 0}
    );
    CSRMatrix<scalar, localIdx> mtx(Vector<scalar>(exec, vals), sparsity);
    Vector<scalar> bValues(exec, {});
    COOMatrix<scalar, localIdx> bMtx(bValues, bSparsity);
    Vector<scalar> bRhs(exec, {});
    return LinearSystem<scalar>(mtx, Vector<scalar>(exec, rhs), bMtx, bMtx, bRhs);
}

std::string multigridJson(const std::string& criteria)
{
    return R"({"type": "solver::Multigrid",
        "mg_level": ["neon::pgmMerge2"],
        "pre_smoother": [{"type": "solver::Ir",
                          "solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
                          "relaxation_factor": 0.6666666666666666,
                          "criteria": [{"type": "Iteration", "max_iters": 2}]}],
        "post_smoother": [{"type": "solver::Ir",
                           "solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
                           "relaxation_factor": 0.6666666666666666,
                           "criteria": [{"type": "Iteration", "max_iters": 2}]}],
        "coarsest_solver": [{"type": "solver::Cg",
                             "criteria": [{"type": "Iteration", "max_iters": 50}]}],
        "min_coarse_rows": 8,
        "criteria": [)"
         + criteria + "]}";
}

std::string writeConfig(const std::string& name, const std::string& json)
{
    auto path = std::filesystem::temp_directory_path() / ("neon_mpgm_" + name + ".json");
    std::ofstream out(path);
    out << json;
    return path.string();
}

scalar maxError(const Vector<scalar>& x)
{
    auto host = x.copyToHost();
    scalar err = 0.0;
    for (auto v : host.view())
        err = std::max(err, std::abs(v - 1.0));
    return err;
}

}

TEST_CASE("MergedPgm generate_reuse in GinkgoSolver - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto reuseSetup = GENERATE(true, false);

    auto cgMg = writeConfig(
        "cg",
        R"({"type": "solver::Cg",
            "criteria": [{"type": "Iteration", "max_iters": 300},
                         {"type": "ResidualNorm", "baseline": "rhs_norm",
                          "reduction_factor": 1e-10}],
            "preconditioner": )"
            + multigridJson(R"({"type": "Iteration", "max_iters": 1})") + "}"
    );
    auto mg = writeConfig("mg", multigridJson(R"({"type": "Iteration", "max_iters": 300},
                         {"type": "ResidualNorm", "baseline": "rhs_norm",
                          "reduction_factor": 1e-10})"));

    for (const auto& config : {cgMg, mg})
    {
        SECTION(execName + " " + config + (reuseSetup ? " reuse" : " no reuse"))
        {
            auto solver = NeoN::la::Solver(
                exec,
                Dictionary {
                    {{"solver", std::string {"Ginkgo"}},
                     {"configFile", config},
                     {"reuseSetup", reuseSetup}}
                }
            );
            // same sparsity with changed values reuses the setup, a different size starts anew
            for (auto [n, d] : {std::pair {300, 2.5}, std::pair {300, 3.0}, std::pair {150, 2.8}})
            {
                auto sys = laplacianSystem(exec, n, d);
                Vector<scalar> x(exec, n, 0.0);
                solver.solve(sys, x);
                REQUIRE(maxError(x) < 1e-6);
            }
        }
    }
}

TEST_CASE("MergedPgm generate_reuse - Ginkgo")
{
    auto exec = gko::ReferenceExecutor::create();
    auto factory = NeoN::la::ginkgo::makeMergedPgmFactory<double>(exec, 2);
    auto A = laplacian1D(exec, 200);

    SECTION("first generate_reuse matches generate")
    {
        auto plain = factory->generate(A);
        auto reuseData = factory->create_empty_reuse_data();
        auto reused = factory->generate_reuse(A, *reuseData);

        REQUIRE(reused->get_coarse_op()->get_size() == plain->get_coarse_op()->get_size());
        REQUIRE(values(reused->get_coarse_op().get()) == values(plain->get_coarse_op().get()));
    }

    SECTION("later generate_reuse keeps the transfer operators and refreshes the coarse op")
    {
        auto reuseData = factory->create_empty_reuse_data();
        auto first = factory->generate_reuse(A, *reuseData);
        auto firstCoarse = values(first->get_coarse_op().get());

        auto A2 = gko::share(gko::clone(A));
        A2->scale(gko::initialize<Dense>({2.0}, exec));
        auto second = factory->generate_reuse(A2, *reuseData);

        REQUIRE(second->get_prolong_op() == first->get_prolong_op());
        REQUIRE(second->get_restrict_op() == first->get_restrict_op());
        auto secondCoarse = values(second->get_coarse_op().get());
        REQUIRE(secondCoarse.size() == firstCoarse.size());
        for (std::size_t i = 0; i < firstCoarse.size(); ++i)
        {
            REQUIRE(secondCoarse[i] == Catch::Approx(2.0 * firstCoarse[i]));
        }
        // the first result is independent of later generations
        REQUIRE(values(first->get_coarse_op().get()) == firstCoarse);
    }

    SECTION("generate_reuse rejects a matrix of a different size")
    {
        auto reuseData = factory->create_empty_reuse_data();
        factory->generate_reuse(A, *reuseData);
        REQUIRE_THROWS_AS(
            factory->generate_reuse(laplacian1D(exec, 100), *reuseData), gko::DimensionMismatch
        );
    }

    SECTION("Multigrid preconditioner reused across matrices")
    {
        auto mg = gko::solver::Multigrid::build()
                      .with_mg_level(factory)
                      .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                      .with_min_coarse_rows(4u)
                      .on(exec);
        auto mgReuse = mg->create_empty_reuse_data();

        for (double d : {2.0, 2.5})
        {
            auto Ad = laplacian1D(exec, 200, d);
            auto precond = gko::share(mg->generate_reuse(Ad, *mgReuse));
            auto solver =
                gko::solver::Cg<double>::build()
                    .with_generated_preconditioner(precond)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(200u),
                        gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-10)
                    )
                    .on(exec)
                    ->generate(Ad);
            auto b = Dense::create(exec, gko::dim<2> {200, 1});
            b->fill(1.0);
            auto x = Dense::create(exec, gko::dim<2> {200, 1});
            x->fill(0.0);
            solver->apply(b, x);

            auto res = gko::clone(b);
            auto one = gko::initialize<Dense>({1.0}, exec);
            auto negOne = gko::initialize<Dense>({-1.0}, exec);
            Ad->apply(negOne, x, one, res);
            auto norm = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm);
            REQUIRE(norm->at(0) < 1e-8 * std::sqrt(200.0));
        }
    }
}

#endif
