// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <cstdlib>

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
    // Ginkgo's distributed components (Matrix, Schwarz preconditioner, …) default
    // their global index type to gko::int64. NeoN's `label` is int32 in the default
    // build, so using <scalar, label, label> for the distributed matrix produces
    // <double, int, int>, which mismatches the Schwarz factory built from the
    // dictionary (<double, int, long long>) and aborts at gko::as<...>() inside
    // Schwarz::generate. Use gko::int64 for the global index throughout to align
    // with Schwarz's default instantiation.
    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;
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
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    // recv_connections, ie the send_idxs of the neighbouring ranks in global indexing.
    // The COO sparsity stores them as IndexType (= label = int32). Cast up to
    // global_index_type (int64) because that's what the index_map / partition / matrix
    // template arguments require above.
    auto bmtxv = bmtx.sparsity()->colIdxs();
    const auto nRecv = static_cast<gko::size_type>(bmtxv.size());

    gko::array<global_index_type> recv_connections {exec, nRecv};
    {
        // Stage on host: the cast from int32 -> int64 is a sequential scalar
        // operation done once per solve setup; size is the number of proc-boundary
        // off-diagonal entries on this rank.
        auto host = exec->get_master();
        auto srcView = gko::array<IndexType>::const_view(exec, nRecv, bmtxv.data());
        auto srcArr = srcView.copy_to_array();
        srcArr.set_executor(host);
        gko::array<global_index_type> hostRecv {host, nRecv};
        auto srcPtr = srcArr.get_const_data();
        auto dstPtr = hostRecv.get_data();
        for (gko::size_type i = 0; i < nRecv; ++i)
        {
            dstPtr[i] = static_cast<global_index_type>(srcPtr[i]);
        }
        recv_connections = std::move(hostRecv);
        recv_connections.set_executor(exec);
    }

    auto imap = gko::experimental::distributed::index_map<label, global_index_type>(
        exec, partition, comm.rank(), recv_connections
    );

    // numNonLocalElements = unique ghost cells (imap dedup of recv_connections).
    // nRecv = proc-face count, which is >= numNonLocalElements when ghost cells are
    // shared by multiple proc faces (a corner/edge cell adjacent to >1 cut face).
    // The COO non-local matrix has nRecv entries (one per proc face) but only
    // numNonLocalElements distinct columns (one per unique ghost cell).  Using nRecv
    // here ensures every proc-face contribution is included; map_to_local maps
    // duplicate global ghost IDs to the same local non-local column index.
    auto numNonLocalElements = imap.get_non_local_size();

    auto non_loc_vals = gko::array<scalar>::const_view(exec, nRecv, bmtx.values().data());
    auto non_loc_row =
        gko::array<IndexType>::const_view(exec, nRecv, bmtx.sparsity()->rowOffs().data());

    // imap is templated on <label, global_index_type=int64>, so map_to_local expects
    // the input to be a gko::array<global_index_type>. Cast all nRecv colIdxs (int32)
    // to int64 — duplicates correctly fold to the same local non-local index.
    gko::array<global_index_type> non_loc_col {exec, nRecv};
    {
        auto host = exec->get_master();
        gko::array<IndexType> srcHost {host, nRecv};
        host->copy_from(exec.get(), nRecv, bmtx.sparsity()->colIdxs().data(), srcHost.get_data());
        gko::array<global_index_type> hostCol {host, nRecv};
        auto srcPtr = srcHost.get_const_data();
        auto dstPtr = hostCol.get_data();
        for (gko::size_type i = 0; i < nRecv; ++i)
            dstPtr[i] = static_cast<global_index_type>(srcPtr[i]);
        non_loc_col = std::move(hostCol);
        non_loc_col.set_executor(exec);
    }

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
    auto nonLocalMtx = COOMatrix<scalar, localIdx> {nonLocalValues, nonLocalSparsity};

    const CommunicationPattern& commPattern = sys.commPattern();
#if defined(NEON_GINKGO_HOST_STAGE)
    bool forceHostBuffer = true;
#else
    bool forceHostBuffer = false;
#endif
    auto comm = gko::experimental::mpi::communicator(commPattern.env.comm(), forceHostBuffer);
    auto gkoMtx = createGkoMtxDist(exec, comm, mtx, nonLocalMtx, commPattern);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, std::move(solver)));
    setComponent<I>(xcopy, x);
}

/* @brief Runtime check for the host-fallback env var.
 *
 * Env: NEON_GINKGO_HOST_SOLVE=1 forces the distributed Ginkgo solve to run
 * on gko::ReferenceExecutor regardless of the NeoN executor. The rest of
 * NeoN keeps running on the original (GPU) executor; only the Ginkgo
 * solve step trips through host memory:
 *   1. LinearSystem and x are copied device -> host
 *   2. A new Ginkgo solver factory is built on ReferenceExecutor (re-parsed
 *      from the same JSON config so the solver/precond match the device case)
 *   3. solve_impl_dist runs on host
 *   4. x is copied back host -> device
 *
 * This isolates whether the v2.0 GPU distributed bug is in Ginkgo's
 * distributed CUDA apply / halo gather / Coo apply2 (all of which are
 * bypassed by ReferenceExecutor) or upstream of Ginkgo. Performance cost:
 * one device<->host round-trip per solve, plus the entire solve on CPU.
 * Default OFF -- production builds untouched.
 */
inline bool hostSolveFallbackEnabled()
{
    static const bool enabled = std::getenv("NEON_GINKGO_HOST_SOLVE") != nullptr;
    return enabled;
}

/* Scalar variant of the host-fallback distributed solve. */
static SolverStats solveDistHostFallback(
    const gko::config::pnode& config,
    const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys,
    Vector<scalar>& x
)
{
    auto hostExec = gko::ReferenceExecutor::create();
    auto hostSys = sys.copyToHost();
    auto xHost = x.copyToHost();

    // On host we always want host buffers; the commPattern.env.comm() is the
    // same MPI communicator the caller used.
    const bool forceHostBuffer = true;
    auto comm =
        gko::experimental::mpi::communicator(hostSys.commPattern().env.comm(), forceHostBuffer);

    auto gkoMtx = createGkoMtxDist(
        hostExec, comm, hostSys.matrix(), hostSys.nonLocalMatrix(), hostSys.commPattern()
    );

    auto hostFactory =
        gko::config::parse(
            config, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
        )
            .on(hostExec);
    auto hostSolver = hostFactory->generate(gkoMtx);

    auto stat =
        solve_impl_dist(hostExec, comm, hostSys.rhs(), xHost, gkoMtx, std::move(hostSolver));

    // Write the solved xHost back into x's device storage.
    x = Vector<scalar>(x.exec(), xHost);

    return SolverStats {{stat}};
}

/* One component of the Vec3 host-fallback solve. Mirrors solveComponentDist
 * but builds its own host-side factory and runs on ReferenceExecutor. */
template<unsigned int I>
static void solveComponentDistHost(
    const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& hostSys,
    Vector<Vec3>& xHost,
    std::shared_ptr<gko::Executor> hostExec,
    const gko::config::pnode& config,
    SolverStats& stats
)
{
    auto rhs = getComponent<I>(hostSys.rhs());
    auto xcopy = getComponent<I>(xHost);
    auto values = getComponent<I>(hostSys.matrix().values());
    auto sparsity = hostSys.matrix().sparsity();
    auto mtx = CSRMatrix<scalar, localIdx> {values, sparsity};

    auto nonLocalValues = getComponent<I>(hostSys.nonLocalMatrix().values());
    auto nonLocalSparsity = hostSys.nonLocalMatrix().sparsity();
    auto nonLocalMtx = COOMatrix<scalar, localIdx> {nonLocalValues, nonLocalSparsity};

    const bool forceHostBuffer = true;
    auto comm =
        gko::experimental::mpi::communicator(hostSys.commPattern().env.comm(), forceHostBuffer);
    auto gkoMtx = createGkoMtxDist(hostExec, comm, mtx, nonLocalMtx, hostSys.commPattern());

    auto hostFactory =
        gko::config::parse(
            config, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
        )
            .on(hostExec);
    auto hostSolver = hostFactory->generate(gkoMtx);

    stats.entries.push_back(
        solve_impl_dist(hostExec, comm, rhs, xcopy, gkoMtx, std::move(hostSolver))
    );
    setComponent<I>(xcopy, xHost);
}

/* Vec3 variant of the host-fallback distributed solve. */
static SolverStats solveDistHostFallback(
    const gko::config::pnode& config,
    const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& sys,
    Vector<Vec3>& x
)
{
    auto hostExec = gko::ReferenceExecutor::create();
    auto hostSys = sys.copyToHost();
    auto xHost = x.copyToHost();

    auto stats = SolverStats {};
    solveComponentDistHost<0>(hostSys, xHost, hostExec, config, stats);
    solveComponentDistHost<1>(hostSys, xHost, hostExec, config, stats);
    solveComponentDistHost<2>(hostSys, xHost, hostExec, config, stats);

    x = Vector<Vec3>(x.exec(), xHost);
    return stats;
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    // NEON_GINKGO_HOST_SOLVE: route the Ginkgo solve through ReferenceExecutor
    // for diagnosis of v2.0 multi-GPU divergence. Keeps NeoN on its native
    // (GPU) executor for everything except the solve.
    if (hostSolveFallbackEnabled() && !std::holds_alternative<SerialExecutor>(x.exec()))
    {
        return solveDistHostFallback(config_, sys, x);
    }

#if defined(NEON_GINKGO_HOST_STAGE)
    bool forceHostBuffer = true;
#else
    bool forceHostBuffer = false;
#endif
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
    if (hostSolveFallbackEnabled() && !std::holds_alternative<SerialExecutor>(x.exec()))
    {
        return solveDistHostFallback(config_, sys, x);
    }

    auto stats = SolverStats {};
    solveComponentDist<0>(sys, x, gkoExec_, factory_, stats);
    solveComponentDist<1>(sys, x, gkoExec_, factory_, stats);
    solveComponentDist<2>(sys, x, gkoExec_, factory_, stats);
    return stats;
}

}

#endif
