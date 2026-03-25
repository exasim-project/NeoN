// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

namespace NeoN::la::ginkgo
{

/*@brief create a dense const view into data given by ptr*/
// std::shared_ptr<const gko::matrix::Dense<scalar>>
std::shared_ptr<gko::LinOp> gkoVecView(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const scalar* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;

    auto size = static_cast<std::size_t>(s);

    auto ret = gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(
            exec,
            gko::dim<2> {size, 1},
            gko::make_const_array_view(exec, size, ptr).copy_to_array(),
            1
        )
    ));
    // gkoArrayView(exec, std::span {ptr, size}), 1)));
    // gko::array<scalar>::const_view(exec, size, ptr), 1)));

    return ret;
    // return gko::share(gko::matrix::Dense<scalar>::create_const(
    //     exec, gko::dim<2> {size, 1}, gko::array<scalar>::const_view(exec, size, ptr), 1
    // ));
}

void writeToDisk(std::string fn, std::shared_ptr<gko::LinOp> A)
{
    std::ofstream stream {fn};
    stream << std::setprecision(15);
    gko::write(stream, A.get());
}

/* @brief create a ginkgo csr matrix by creating views into Csr<scalar> avoiding copies */
template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx, //, local mtx
    const CSRMatrix<scalar, IndexType>& bmtx //, local mtx
    // const LinearSystem<scalar, IndexType>& ls
)
{
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, label>;
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    // FIXME currently ls keeps a local mtx
    // FIXME currently only mtx.local() is used
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

    // FIXME dont recreate
    auto partition =
        gko::share(gko::experimental::distributed::build_partition_from_local_size<label, label>(
            exec, comm, nrows
        ));

    // FIXME currently no communication with other rank
    // recv_connections, ie the send_idxs of the neighbouring ranks in global indexing
    auto recv_connections = gko::array<label>(exec, 0);

    auto imap = gko::experimental::distributed::index_map<label, label>(
        exec, partition, comm.rank(), recv_connections
    );

    // FIXME why cloned?
    std::shared_ptr<gko::LinOp> localMtx =
        gko::share(
            gko::matrix::Csr<scalar, IndexType>::create_const(
                exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
            )
        )
            ->clone();


    auto non_loc_vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), bmtx.values().data()
    );
    auto non_loc_col = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->colIdxs().size()),
        bmtx.sparsity()->colIdxs().data()
    );
    auto non_loc_row = gko::array<IndexType>::const_view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->rowOffs().size()),
        bmtx.sparsity()->rowOffs().data()
    );


    writeToDisk("localA" + std::to_string(comm.rank()) + ".mtx", localMtx);
    std::shared_ptr<gko::LinOp> nonLocalMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create(exec, gko::dim<2> {nrows, 0}));
    writeToDisk("nonLocalA" + std::to_string(comm.rank()) + ".mtx", nonLocalMtx);

    return gko::share(dist_mtx::create(exec, comm, imap, localMtx, nonLocalMtx));
}

SolverStatsEntry solve_impl_dist(
    std::shared_ptr<const gko::Executor> exec,
    const Vector<scalar>& rhs,
    Vector<scalar>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver
)
{
    exec->synchronize();

    // FIXME dont re-init
    bool forceHostBuffer = false;
    mpi::Environment env;
    auto comm = gko::experimental::mpi::communicator(env.comm(), forceHostBuffer);

    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, comm, rhs.data(), nrows);
    auto x = gkoVecView(exec, comm, xIn.data(), nrows);

    // create a copy of rhs so that we can inline compute
    // the residual
    auto rhsCopy = Vector<scalar>(rhs);
    auto res = gkoVecView(exec, comm, rhsCopy.data(), nrows);

    // compute Ax-b -> res
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // FIXME dont re-init
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

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    // TODO make that selectable via dictionary
    bool forceHostBuffer = false;
    mpi::Environment env;
    auto comm = gko::experimental::mpi::communicator(env.comm(), forceHostBuffer);
    auto gkoMtx = createGkoMtx(gkoExec_, comm, sys.matrix(), sys.boundaryMatrix());

    auto solver = factory_->generate(gkoMtx);
    return {solve_impl_dist(gkoExec_, sys.rhs(), x, gkoMtx, std::move(solver))};
}
}

#endif
