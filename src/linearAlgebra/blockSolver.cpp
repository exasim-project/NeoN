// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <chrono>

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoN/linearAlgebra/blockSolver.hpp"
#include "NeoN/linearAlgebra/ginkgo.hpp"

namespace NeoN::la
{

BlockSolver::BlockSolver(const Executor& exec, const Dictionary& solverDict)
    : exec_(exec), solverDict_(solverDict)
{}

SolverStats
BlockSolver::solve(const BlockMatrix& matrix, const BlockVector& rhs, BlockVector& solution) const
{
    auto gkoExec = ginkgo::getGkoExecutor(exec_);

    // -- Zero-copy CSR matrix from BlockMatrix --
    const auto& bm = matrix.blockCSRMatrix();
    auto blockSp = bm.sparsity(); // shared_ptr<const BlockSparsityPattern>

    auto nrows = static_cast<gko::size_type>(blockSp->rows());
    auto nnz = static_cast<gko::size_type>(blockSp->nnz());
    auto nRowOffs = static_cast<gko::size_type>(blockSp->rowOffs().size());

    auto vals = gko::array<scalar>::const_view(gkoExec, nnz, bm.values().data());
    auto col = gko::array<localIdx>::const_view(gkoExec, nnz, blockSp->colIdxs().data());
    auto row = gko::array<localIdx>::const_view(gkoExec, nRowOffs, blockSp->rowOffs().data());

    auto gkoMtx = gko::share(gko::matrix::Csr<scalar, localIdx>::create_const(
        gkoExec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
    ));

    // -- Zero-copy dense vectors from BlockVector --
    auto totalSize = static_cast<gko::size_type>(rhs.totalSize());

    auto b = gko::share(gko::matrix::Dense<scalar>::create_const(
        gkoExec,
        gko::dim<2> {totalSize, 1},
        gko::array<scalar>::const_view(gkoExec, totalSize, rhs.vector().data()),
        1
    ));

    auto x = gko::share(gko::matrix::Dense<scalar>::create(
        gkoExec,
        gko::dim<2> {totalSize, 1},
        gko::make_array_view(gkoExec, totalSize, solution.vector().data()),
        1
    ));

    // -- Create solver factory from config --
    using vec = gko::matrix::Dense<scalar>;
    auto config = ginkgo::parse(solverDict_);
    auto factory = gko::config::parse(
                       config, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
    )
                       .on(gkoExec);

    gkoExec->synchronize();
    auto startEval = std::chrono::steady_clock::now();

    // -- Compute initial residual --
    auto rhsCopy = Vector<scalar>(rhs.vector());
    auto res = gko::share(gko::matrix::Dense<scalar>::create(
        gkoExec,
        gko::dim<2> {totalSize, 1},
        gko::make_array_view(gkoExec, totalSize, rhsCopy.data()),
        1
    ));

    auto one = gko::initialize<vec>({1.0}, gkoExec);
    auto neg_one = gko::initialize<vec>({-1.0}, gkoExec);
    gkoMtx->apply(one, x, neg_one, res);

    auto init = gko::initialize<vec>({0.0}, gkoExec);
    res->compute_norm2(init);
    auto host = vec::create(gkoExec->get_master(), gko::dim<2> {1});
    scalar initResNorm = host->copy_from(init)->at(0);

    // -- Solve --
    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    auto solver = factory->generate(gkoMtx);
    solver->add_logger(logger);
    solver->apply(b, x);

    scalar finalResNorm = host->copy_from(gko::as<vec>(logger->get_residual_norm()))->at(0);
    auto numIter = static_cast<int>(logger->get_num_iterations());

    gkoExec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    return SolverStats {numIter, initResNorm, finalResNorm, duration};
}

} // namespace NeoN::la

#endif
