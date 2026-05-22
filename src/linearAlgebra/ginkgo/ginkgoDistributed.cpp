// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO
#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

namespace NeoN::la::ginkgo
{

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
    return gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(exec, gko::dim<2> {size, 1}, gko::array<scalar>::view(exec, size, ptr), 1)
    ));
}

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
    return gko::share(dist_vec::create_const(
        exec,
        comm,
        vec::create_const(
            exec, gko::dim<2> {size, 1}, gko::array<scalar>::const_view(exec, size, ptr), 1
        )
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtxDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx,
    const COOMatrix<scalar, IndexType>& bmtx,
    const CommunicationPattern& commPattern
)
{
    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;

    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(mtx.values().size()), mtx.values().data()
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

    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    // recv_connections: global cell indices of neighbor cells (stored in bmtx colIdxs),
    // cast to int64 as required by gko::experimental::distributed::index_map.
    const auto& bmtxColIdxs = bmtx.sparsity()->colIdxs();
    const auto nRecv = static_cast<gko::size_type>(bmtxColIdxs.size());

    gko::array<global_index_type> recv_connections {exec, nRecv};
    {
        auto host = exec->get_master();
        auto srcView = gko::array<IndexType>::const_view(exec, nRecv, bmtxColIdxs.data());
        auto srcArr = srcView.copy_to_array();
        srcArr.set_executor(host);
        gko::array<global_index_type> hostRecv {host, nRecv};
        const auto* srcPtr = srcArr.get_const_data();
        auto* dstPtr = hostRecv.get_data();
        for (gko::size_type i = 0; i < nRecv; ++i)
            dstPtr[i] = static_cast<global_index_type>(srcPtr[i]);
        recv_connections = std::move(hostRecv);
        recv_connections.set_executor(exec);
    }

    auto imap = gko::experimental::distributed::index_map<label, global_index_type>(
        exec, partition, comm.rank(), recv_connections
    );

    const auto numNonLocalElements = imap.get_non_local_size();

    auto non_loc_vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(numNonLocalElements), bmtx.values().data()
    );
    // rowIdxs() holds global row indices; convert to local (subtract this rank's global offset).
    const auto globalOffset = static_cast<IndexType>(partition->get_range_bounds()[comm.rank()]);
    gko::array<IndexType> non_loc_row {exec, static_cast<gko::size_type>(numNonLocalElements)};
    {
        auto host = exec->get_master();
        auto srcView = gko::array<IndexType>::const_view(
            exec,
            static_cast<gko::size_type>(numNonLocalElements),
            bmtx.sparsity()->rowIdxs().data()
        );
        auto srcArr = srcView.copy_to_array();
        srcArr.set_executor(host);
        gko::array<IndexType> hostRow {host, static_cast<gko::size_type>(numNonLocalElements)};
        const auto* srcPtr = srcArr.get_const_data();
        auto* dstPtr = hostRow.get_data();
        for (gko::size_type i = 0; i < static_cast<gko::size_type>(numNonLocalElements); ++i)
            dstPtr[i] = static_cast<IndexType>(srcPtr[i]) - globalOffset;
        non_loc_row = std::move(hostRow);
        non_loc_row.set_executor(exec);
    }

    // Cast colIdxs (global neighbor indices, int32) to int64 for index_map::map_to_local
    gko::array<global_index_type> non_loc_col {
        exec, static_cast<gko::size_type>(numNonLocalElements)
    };
    {
        auto host = exec->get_master();
        auto srcView = gko::array<IndexType>::const_view(
            exec,
            static_cast<gko::size_type>(numNonLocalElements),
            bmtx.sparsity()->colIdxs().data()
        );
        auto srcArr = srcView.copy_to_array();
        srcArr.set_executor(host);
        gko::array<global_index_type> hostCol {
            host, static_cast<gko::size_type>(numNonLocalElements)
        };
        const auto* srcPtr = srcArr.get_const_data();
        auto* dstPtr = hostCol.get_data();
        for (gko::size_type i = 0; i < static_cast<gko::size_type>(numNonLocalElements); ++i)
            dstPtr[i] = static_cast<global_index_type>(srcPtr[i]);
        non_loc_col = std::move(hostCol);
        non_loc_col.set_executor(exec);
    }

    auto comp_non_loc_col =
        imap.map_to_local(non_loc_col, gko::experimental::distributed::index_space::non_local);

    auto nonLocalMtx = gko::share(gko::matrix::Coo<scalar, IndexType>::create_const(
                                      exec,
                                      gko::dim<2> {nrows, numNonLocalElements},
                                      std::move(non_loc_vals),
                                      comp_non_loc_col.as_const_view(),
                                      non_loc_row.as_const_view()
    )
                                      ->clone());

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

    auto rhsCopy = Vector<scalar>(rhs);
    auto res = gkoVecViewDist(exec, comm, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    auto init = gko::initialize<vec>({0.0}, exec);
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    gko::as<dist_vec>(res)->compute_norm2(init);
    scalar initResNorm = retrieve(init);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

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

    auto nonLocalValues = getComponent<I>(sys.offDiagonalMatrix().values());
    auto nonLocalSparsity = sys.offDiagonalMatrix().sparsity();
    auto nonLocalMtx = COOMatrix<scalar, localIdx> {nonLocalValues, nonLocalSparsity};

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
    bool forceHostBuffer = false;
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(commPattern.env.comm(), forceHostBuffer);
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);
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

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
