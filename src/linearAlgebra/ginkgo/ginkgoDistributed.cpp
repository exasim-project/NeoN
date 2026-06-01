// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO
#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/error.hpp"

#include <memory>
#include <vector>

#include "NeoN/core/parallelAlgorithms.hpp"

#include <algorithm>
#include <numeric>

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
    // commPattern is currently unused here: all the connectivity information needed to build
    // the distributed matrix is already encoded in the row/column indices of `mtx` (local block)
    // and `bmtx` (off-diagonal/processor coupling).
    static_cast<void>(commPattern);

    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;

    // Local block: zero-copy CSR views over the existing NeoN storage. The local matrix is by far
    // the largest part and is reused as-is on every solve, so it is never copied/re-expanded here.
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

    const auto nrows = static_cast<gko::size_type>(mtx.sparsity()->rows());

    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    // First global row index owned by this rank. get_range_bounds() points into the partition's
    // executor memory (device memory when `exec` is a GPU executor), so the single value is pulled
    // off the device safely rather than dereferenced directly on the host.
    const auto globalOffset = exec->copy_val_to_host(partition->get_range_bounds() + comm.rank());

    // Off-diagonal block: rowIdxs()/colIdxs() hold global indices (owning cell's global row and the
    // neighbour cell's global column). Only this (small) block is processed per solve.
    const auto nNonLocalNnz = static_cast<gko::size_type>(bmtx.values().size());
    auto host = exec->get_master();

    // FIXME Do this on device
    const auto bColH = bmtx.sparsity()->colIdxs().copyToHost();
    const auto bRowH = bmtx.sparsity()->rowIdxs().copyToHost();
    const auto bValH = bmtx.values().copyToHost();

    // recv_connections are the global neighbour-column indices; index_map deduplicates and keeps
    // only those it owns remotely, defining the non-local column space.
    gko::array<global_index_type> recv_connections {host, nNonLocalNnz};
    {
        auto* dst = recv_connections.get_data();
        const auto* src = bColH.data();
        for (gko::size_type i = 0; i < nNonLocalNnz; ++i)
            dst[i] = static_cast<global_index_type>(src[i]);
    }
    recv_connections.set_executor(exec);

    auto imap = gko::experimental::distributed::index_map<label, global_index_type>(
        exec, partition, comm.rank(), recv_connections
    );
    const auto numNonLocalElements = imap.get_non_local_size();

    // Map every off-diagonal column into the non-local index space. A column the index_map does not
    // classify as a known remote column (e.g. one that resolves into this rank's own range) maps to
    // invalid_index. Such a coupling is already represented in the local block, so it is dropped
    // here rather than fed into the COO, where invalid_index would become an out-of-bounds column.
    auto mapped =
        imap.map_to_local(recv_connections, gko::experimental::distributed::index_space::non_local);
    const auto mappedH = gko::array<label>(host, mapped);
    const auto* mappedPtr = mappedH.get_const_data();
    const auto* rowPtr = bRowH.data();
    const auto* valPtr = bValH.data();

    gko::array<IndexType> nlCol {host, nNonLocalNnz};
    gko::array<IndexType> nlRow {host, nNonLocalNnz};
    gko::array<scalar> nlVal {host, nNonLocalNnz};
    auto* nlColPtr = nlCol.get_data();
    auto* nlRowPtr = nlRow.get_data();
    auto* nlValPtr = nlVal.get_data();

    constexpr auto invalid = gko::invalid_index<label>();
    gko::size_type kept = 0;
    for (gko::size_type i = 0; i < nNonLocalNnz; ++i)
    {
        if (mappedPtr[i] == invalid) continue;
        nlColPtr[kept] = static_cast<IndexType>(mappedPtr[i]);
        nlRowPtr[kept] =
            static_cast<IndexType>(static_cast<global_index_type>(rowPtr[i]) - globalOffset);
        nlValPtr[kept] = valPtr[i];
        ++kept;
    }

    // Sort COO by row for Ginkgo's CUDA Coo::apply2: its warp segmented-scan incorrectly reduces
    // non-contiguous same-row entries, making the GPU non-local apply non-symmetric. The
    // Reference/CPU apply2 is order-robust. All data is on the host here, so we sort in-place.
    std::vector<gko::size_type> perm(kept);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(
        perm.begin(), perm.end(), [&](gko::size_type a, gko::size_type b)
        { return nlRowPtr[a] < nlRowPtr[b]; }
    );
    gko::array<IndexType> sRow {host, kept};
    gko::array<IndexType> sCol {host, kept};
    gko::array<scalar> sVal {host, kept};
    for (gko::size_type i = 0; i < kept; ++i)
    {
        sRow.get_data()[i] = nlRowPtr[perm[i]];
        sCol.get_data()[i] = nlColPtr[perm[i]];
        sVal.get_data()[i] = nlValPtr[perm[i]];
    }
    sRow.set_executor(exec);
    sCol.set_executor(exec);
    sVal.set_executor(exec);

    auto nonLocalMtx =
        gko::share(gko::matrix::Coo<scalar, IndexType>::create_const(
                       exec,
                       gko::dim<2> {nrows, numNonLocalElements},
                       gko::array<scalar>::const_view(exec, kept, sVal.get_const_data()),
                       gko::array<IndexType>::const_view(exec, kept, sCol.get_const_data()),
                       gko::array<IndexType>::const_view(exec, kept, sRow.get_const_data())
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
    std::unique_ptr<gko::LinOp> solver,
    const L1ResidualControl* l1Control = nullptr
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();
    using vec = gko::matrix::Dense<scalar>;
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    label nrows = rhs.size();

    const auto b = gkoConstVecViewDist(exec, comm, rhs.data(), nrows);
    auto x = gkoVecViewDist(exec, comm, xIn.data(), nrows);

    // L1-scaled residual path: stop and report on the (globally reduced) scaled residual.
    if (l1Control != nullptr)
    {
        auto l1Res = solveWithL1StopDist(
            exec,
            mtx,
            std::dynamic_pointer_cast<const dist_vec>(b),
            std::dynamic_pointer_cast<dist_vec>(x),
            solver.get(),
            *l1Control
        );
        exec->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;
        return {static_cast<label>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration};
    }

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

    auto rhsCopyFinal = Vector<scalar>(rhs);
    auto resFinal = gkoVecViewDist(exec, comm, rhsCopyFinal.data(), nrows);
    mtx->apply(one, x, neg_one, resFinal);
    auto finalNormVec = gko::initialize<vec>({0.0}, exec);
    gko::as<dist_vec>(resFinal)->compute_norm2(finalNormVec);
    scalar finalResNorm = retrieve(finalNormVec);

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
void solveComponentDist(
    auto& sys, auto& x, auto& exec, auto& factory, auto& stats, const L1ResidualControl* l1Control
)
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
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx = createGkoMtxDist(exec, comm, mtx, nonLocalMtx, commPattern);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(
        solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, std::move(solver), l1Control)
    );
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);
    auto solver = factory_->generate(gkoMtx);
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, std::move(solver), l1Control)};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    solveComponentDist<0>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<1>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<2>(sys, x, gkoExec_, factory_, stats, l1Control);
    return stats;
}

// Solve one Vec3 rhs component against a shared distributed scalar matrix (segregated form).
template<unsigned int I>
void solveVec3RhsComponentDist(
    const Vector<Vec3>& rhs,
    Vector<Vec3>& x,
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    std::shared_ptr<const gko::LinOpFactory> factory,
    std::shared_ptr<const gko::LinOp> gkoMtx,
    SolverStats& stats,
    const L1ResidualControl* l1Control
)
{
    auto rhsComp = getComponent<I>(rhs);
    auto xcopy = getComponent<I>(x);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(
        solve_impl_dist(exec, comm, rhsComp, xcopy, gkoMtx, std::move(solver), l1Control)
    );
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>& sys,
    Vector<Vec3>& x
) const
{
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    // The matrix is already scalar and shared across all three components, so build the
    // distributed operator once and reuse it for each Vec3 rhs component.
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);

    // TODO here Ginkgos multiple RHS solver could be used
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    auto stats = SolverStats {};
    solveVec3RhsComponentDist<0>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponentDist<1>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponentDist<2>(sys.rhs(), x, gkoExec_, comm, factory_, gkoMtx, stats, l1Control);
    return stats;
}

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
