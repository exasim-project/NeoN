// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

namespace NeoN::la::ginkgo
{

/** @brief create a dense const view into data given by ptr*/
std::shared_ptr<gko::LinOp> gkoVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    auto size = static_cast<std::size_t>(s);

    auto ret = gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(exec, gko::dim<2> {size, 1}, gko::array<scalar>::view(exec, size, ptr), 1)
    ));

    return ret;
}

/** @brief create a dense const view into data given by ptr*/
std::shared_ptr<const gko::LinOp> gkoConstVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    auto size = static_cast<std::size_t>(s);

    auto ret = gko::share(dist_vec::create_const(
        exec,
        comm,
        vec::create_const(
            exec, gko::dim<2> {size, 1}, gko::array<scalar>::const_view(exec, size, ptr), 1
        )
    ));

    return ret;
}


void writeToDisk(std::string fn, std::shared_ptr<const gko::matrix::Coo<scalar, localIdx>> A)
{
    std::ofstream stream {fn};
    stream << std::setprecision(15);
    gko::write(stream, A.get());
}

/* @brief create a ginkgo csr matrix by creating views into Csr<scalar> avoiding copies */
template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtxDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx,
    const COOMatrix<scalar, IndexType>& bmtx,
    const CommunicationPattern& commPattern
)
{
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), mtx.values().data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->colIdxs().size()),
        mtx.sparsity()->colIdxs().data()
    );
    auto row = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->rowOffs().size()),
        mtx.sparsity()->rowOffs().data()
    );

    auto nrows = static_cast<gko::size_type>(mtx.sparsity()->rows());

    // TODO dont recreate
    auto partition =
        gko::share(gko::experimental::distributed::build_partition_from_local_size<label, label>(
            exec, comm, nrows
        ));

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    // Non local part of matrix
    auto numNonLocalElements = commPattern.sendCounts[commPattern.sendCounts.size() - 1];

    // recv_connections, ie the send_idxs of the neighbouring ranks in global indexing
    auto bmtxv = bmtx.sparsity()->colIdxs();

    gko::array<int> recv_connections = gko::make_array_view(exec, bmtxv.size(), bmtxv.data());

    auto imap = gko::experimental::distributed::index_map<label, label>(
        exec, partition, comm.rank(), recv_connections
    );

    auto non_loc_vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(numNonLocalElements), bmtx.values().data()
    );
    auto non_loc_row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(numNonLocalElements), bmtx.sparsity()->rowOffs().data()
    );

    auto non_loc_col = gko::array<IndexType>::const_view(
                           exec,
                           static_cast<gko::size_type>(numNonLocalElements),
                           bmtx.sparsity()->colIdxs().data()
    )
                           .copy_to_array();

    auto comp_non_loc_col =
        imap.map_to_local(non_loc_col, gko::experimental::distributed::index_space::non_local);

    // NOTE currently we copy recompute the non local column indices thus we also clone the matrix
    // here to avoid any dangling pointer
    auto nonLocalMtx = gko::share(gko::matrix::Coo<scalar, IndexType>::create_const(
                                      exec,
                                      gko::dim<2> {nrows, numNonLocalElements},
                                      std::move(non_loc_vals),
                                      comp_non_loc_col.as_const_view(),
                                      std::move(non_loc_row)
    )
                                      ->clone());

    // writeToDisk("localA" + std::to_string(comm.rank()) + ".mtx", localMtx);
    // writeToDisk("nonLocalA" + std::to_string(comm.rank()) + ".mtx", nonLocalMtx);
    // NOTE we need to const_pointer_cast here to cast from const gko::LinOp to LinOp
    return gko::share(dist_mtx::create(
        exec, comm, imap, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtx
    ));
}

SolverStatsEntry solve_impl_dist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const Vector<scalar>& rhs,
    Vector<scalar>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();
    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();

    const auto b = gkoConstVecViewDist(exec, comm, rhs.data(), nrows);
    auto x = gkoVecViewDist(exec, comm, xIn.data(), nrows);

    // create a copy of rhs so that we can inline compute
    // the residual
    auto rhsCopy = Vector<scalar>(rhs);
    auto res = gkoVecViewDist(exec, comm, rhsCopy.data(), nrows);

    // compute Ax-b -> res
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // TODO dont re-init
    auto init = gko::initialize<vec>({0.0}, exec);
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    gko::as<dist_vec>(res)->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    // since we work on a copy we need to copy back
    scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));

    auto numIter = label(logger->get_num_iterations());
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    return {numIter, initResNorm, finalResNorm, duration};
}

template<unsigned int I>
void solveComponentDist(auto& sys, auto& x, auto& exec, auto& factory, auto& stats)
{
    auto rhs = getComponent<I>(sys.rhs());
    auto xcopy = getComponent<I>(x);
    auto values = getComponent<I>(sys.matrix().values());
    auto sparsity = sys.matrix().sparsity();
    auto mtx = CSRMatrix<scalar, localIdx> {values, sparsity};

    auto nonLocalValues = getComponent<I>(sys.nonLocalMatrix().values());
    auto nonLocalSparsity = sys.nonLocalMatrix().sparsity();
    auto nonLocalMtx = COOMatrix<scalar, localIdx> {values, nonLocalSparsity};

    const CommunicationPattern& commPattern = sys.commPattern();
    bool forceHostBuffer = false;
    auto comm = gko::experimental::mpi::communicator(commPattern.env.comm(), forceHostBuffer);
    auto gkoMtx = createGkoMtxDist(exec, comm, mtx, nonLocalMtx, commPattern);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, std::move(solver)));
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    // TODO make that selectable via dictionary
    bool forceHostBuffer = false;
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(commPattern.env.comm(), forceHostBuffer);
    auto gkoMtx = createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.nonLocalMatrix(), commPattern);
    auto solver = factory_->generate(gkoMtx);
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, std::move(solver))};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    solveComponentDist<0>(sys, x, gkoExec_, factory_, stats);
    solveComponentDist<1>(sys, x, gkoExec_, factory_, stats);
    solveComponentDist<2>(sys, x, gkoExec_, factory_, stats);
    return stats;
}

}

#endif
