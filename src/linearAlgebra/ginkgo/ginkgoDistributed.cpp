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

std::shared_ptr<gko::LinOp> gkoVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    Vec3* ptr,
    localIdx s
)
{
    using dist_vec = gko::experimental::distributed::Vector<scalar>;
    using vec = gko::matrix::Dense<scalar>;
    auto size = static_cast<std::size_t>(s);
    return gko::share(dist_vec::create(
        exec,
        comm,
        vec::create(
            exec,
            gko::dim<2> {size, 3},
            gko::array<scalar>::view(exec, size * 3, reinterpret_cast<scalar*>(ptr)),
            3
        )
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

std::shared_ptr<const gko::LinOp> gkoConstVecViewDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const Vec3* ptr,
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
            exec,
            gko::dim<2> {size, 3},
            gko::array<scalar>::const_view(exec, size * 3, reinterpret_cast<const scalar*>(ptr)),
            3
        )
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtxDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx,
    const COOMatrix<scalar, IndexType>& bmtx,
    const CommunicationPattern& commPattern,
    std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>& imapCache,
    std::shared_ptr<gko::matrix::Coo<scalar, IndexType>>& nonLocalMtxCache
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

    std::shared_ptr<const gko::LinOp> localMtx =
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
        ));

    const auto nNonLocalNnz = static_cast<gko::size_type>(bmtx.values().size());

    if (imapCache && nonLocalMtxCache)
    {
        // Fast path: topology is fixed — only the off-diagonal values change each solve.
        const auto bValV = bmtx.values().view();
        auto* cachedValsPtr = nonLocalMtxCache->get_values();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) { cachedValsPtr[i] = bValV[i]; },
            "updateNonLocalValues"
        );
        fence(bmtx.exec());
        return gko::share(dist_mtx::create(
            exec, comm, *imapCache, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtxCache
        ));
    }

    // First call: build partition, index_map, column mapping, and non-local COO structure.
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    // First global row index owned by this rank. get_range_bounds() points into the partition's
    // executor memory (device memory when `exec` is a GPU executor), so the single value is pulled
    // off the device safely rather than dereferenced directly on the host.
    const auto globalOffset = exec->copy_val_to_host(partition->get_range_bounds() + comm.rank());

    // Off-diagonal block: rowIdxs()/colIdxs() are pre-sorted by ascending faceOwner (local row)
    // from the assembly phase — no host copies or sort are needed here.

    // Widen column indices from IndexType to global_index_type on device.
    Vector<global_index_type> widenedCols(bmtx.exec(), static_cast<localIdx>(nNonLocalNnz));
    {
        auto widenedColsView = widenedCols.view();
        const auto offDiagColsView = bmtx.sparsity()->colIdxs().view();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) {
                widenedColsView[i] = static_cast<global_index_type>(offDiagColsView[i]);
            },
            "widenOffDiagonalColumns"
        );
        fence(bmtx.exec());
    }

    // recv_connections: global neighbour-column indices for index_map construction. Built as a
    // copy of the device-resident widenedCols, avoiding a separate host-side widening loop.
    auto recv_connections =
        gko::array<global_index_type>::const_view(exec, nNonLocalNnz, widenedCols.data())
            .copy_to_array();

    if (!imapCache)
    {
        imapCache =
            std::make_shared<gko::experimental::distributed::index_map<label, global_index_type>>(
                exec, partition, comm.rank(), recv_connections
            );
    }
    const auto& imap = *imapCache;
    const auto numNonLocalElements = imap.get_non_local_size();

    // Map global column indices into the non-local index space. Every off-diagonal entry maps to
    // a known remote column — the assembly phase guarantees this by construction.
    const auto mapped =
        imap.map_to_local(recv_connections, gko::experimental::distributed::index_space::non_local);

    // Build local-row, local-column, and value arrays on device. Row indices are already
    // local (0-based per rank) — no offset subtraction needed.
    Vector<IndexType> nlRow(bmtx.exec(), static_cast<localIdx>(nNonLocalNnz));
    Vector<IndexType> nlCol(bmtx.exec(), static_cast<localIdx>(nNonLocalNnz));
    Vector<scalar> nlVal(bmtx.exec(), static_cast<localIdx>(nNonLocalNnz));
    {
        auto nlRowV = nlRow.view();
        auto nlColV = nlCol.view();
        auto nlValV = nlVal.view();
        const auto bRowV = bmtx.sparsity()->rowIdxs().view();
        const auto bValV = bmtx.values().view();
        const auto* mappedPtr = mapped.get_const_data();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) {
                nlRowV[i] = static_cast<IndexType>(bRowV[i]);
                nlColV[i] = static_cast<IndexType>(mappedPtr[i]);
                nlValV[i] = bValV[i];
            },
            "buildNonLocalCOO"
        );
        fence(bmtx.exec());
    }

    nonLocalMtxCache =
        gko::share(gko::matrix::Coo<scalar, IndexType>::create_const(
                       exec,
                       gko::dim<2> {nrows, numNonLocalElements},
                       gko::array<scalar>::const_view(exec, nNonLocalNnz, nlVal.data()),
                       gko::array<IndexType>::const_view(exec, nNonLocalNnz, nlCol.data()),
                       gko::array<IndexType>::const_view(exec, nNonLocalNnz, nlRow.data())
        )
                       ->clone());

    return gko::share(dist_mtx::create(
        exec, comm, imap, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtxCache
    ));
}

SolverStatsEntry solve_impl_dist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const Vector<scalar>& rhs,
    Vector<scalar>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<gko::LinOp> solver,
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

    // copy of rhs to compute the initial residual (res is modified in-place by apply)
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

    // copy of rhs to compute the final residual (resFinal is modified in-place by apply)
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

SolverStats solve_impl_dist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const Vector<Vec3>& rhs,
    Vector<Vec3>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<gko::LinOp> solver,
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
        // Multi-RHS: return one SolverStatsEntry per column using the per-column L1 norms.
        if (!l1Res.perColInitNorms.empty())
        {
            SolverStats stats;
            for (std::size_t i = 0; i < l1Res.perColInitNorms.size(); ++i)
            {
                stats.entries.push_back(
                    {static_cast<label>(l1Res.numIter),
                     l1Res.perColInitNorms[i],
                     l1Res.perColFinalNorms[i],
                     duration}
                );
            }
            return stats;
        }
        return {static_cast<label>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration};
    }

    auto rhsCopy = Vector<Vec3>(rhs);
    auto res = gkoVecViewDist(exec, comm, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // compute_norm2 on a [n x 3] dist_vec writes a [1 x 3] result — one L2 norm per column.
    auto colNorms = [&](std::shared_ptr<gko::LinOp> v) -> std::array<scalar, 3>
    {
        auto nv = vec::create(exec, gko::dim<2> {1, 3});
        gko::as<dist_vec>(v)->compute_norm2(nv);
        auto nh = vec::create(exec->get_master(), gko::dim<2> {1, 3});
        nh->copy_from(nv);
        return {nh->at(0, 0), nh->at(0, 1), nh->at(0, 2)};
    };
    auto initNorms = colNorms(res);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    // restore rhsCopy to b (in-place deep copy, no reallocation) then reuse for final residual
    rhsCopy = rhs;
    res = gkoVecViewDist(exec, comm, rhsCopy.data(), nrows);
    mtx->apply(one, x, neg_one, res);
    auto finalNorms = colNorms(res);

    auto numIter = label(logger->get_num_iterations());
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    SolverStats stats;
    for (int i = 0; i < 3; ++i)
        stats.entries.push_back({numIter, initNorms[i], finalNorms[i], duration});
    return stats;
}

template<unsigned int I>
void solveComponentDist(
    auto& sys,
    auto& x,
    auto& exec,
    auto& factory,
    auto& stats,
    const L1ResidualControl* l1Control,
    std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>& imapCache,
    std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>& nonLocalMtxCache
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
    auto gkoMtx =
        createGkoMtxDist(exec, comm, mtx, nonLocalMtx, commPattern, imapCache, nonLocalMtxCache);
    auto solver = gko::share(factory->generate(gkoMtx));
    stats.entries.push_back(solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, solver, l1Control));
    setComponent<I>(xcopy, x);
}

// Distributed counterpart of solveImplicitTransformComponent: solve component I of a scalar-matrix
// / Vec3-rhs system under an implicit transform BC (slip/symmetry), temporarily applying the
// component's diagonal correction to the shared rank-local diagonal in place and reusing
// solve_impl_dist (which honours the l1ScaledResidual criterion). The correction is rank-local, so
// only the local diagonal entries are touched.
// NOTE: explicit template parameters (not abbreviated `auto` params): nvcc rejects the extended
// __device__ NEON_LAMBDA parallelFor bodies below when they sit inside an abbreviated function
// template. Mirrors solveImplicitTransformComponent in ginkgo.cpp.
template<
    unsigned int I,
    typename SystemType,
    typename ExecType,
    typename FactoryType,
    typename ValuesType,
    typename MatrixAddressingType,
    typename DiagCType>
void solveImplicitTransformComponentDist(
    const SystemType& sys,
    Vector<Vec3>& x,
    const ExecType& exec,
    std::shared_ptr<const gko::Executor> gkoExec,
    const gko::experimental::mpi::communicator& comm,
    std::shared_ptr<const gko::LinOp> gkoMtx,
    const FactoryType& factory,
    SolverStats& stats,
    const L1ResidualControl* l1Control,
    ValuesType values,
    const MatrixAddressingType& ma,
    DiagCType diagC,
    localIdx nrows
)
{
    parallelFor(
        exec,
        {0, nrows},
        NEON_LAMBDA(const localIdx cell) {
            Kokkos::atomic_sub(&values[ma.diagIdx(cell)], diagC[cell][I]);
        },
        "applyImplicitTransformDiagDist"
    );
    gkoExec->synchronize();

    auto rhs = getComponent<I>(sys.rhs());
    auto xcopy = getComponent<I>(x);
    auto solver = gko::share(factory->generate(gkoMtx));
    stats.entries.push_back(solve_impl_dist(gkoExec, comm, rhs, xcopy, gkoMtx, solver, l1Control));
    setComponent<I>(xcopy, x);

    parallelFor(
        exec,
        {0, nrows},
        NEON_LAMBDA(const localIdx cell) {
            Kokkos::atomic_add(&values[ma.diagIdx(cell)], diagC[cell][I]);
        },
        "restoreImplicitTransformDiagDist"
    );
    gkoExec->synchronize();
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx = createGkoMtxDist(
        gkoExec_,
        comm,
        sys.matrix(),
        sys.offDiagonalMatrix(),
        commPattern,
        cachedImap_,
        cachedNonLocalMtx_
    );
    auto solver = gko::share(factory_->generate(gkoMtx));
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, solver, l1Control)};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    solveComponentDist<0>(
        sys, x, gkoExec_, factory_, stats, l1Control, cachedImap_, cachedNonLocalMtx_
    );
    solveComponentDist<1>(
        sys, x, gkoExec_, factory_, stats, l1Control, cachedImap_, cachedNonLocalMtx_
    );
    solveComponentDist<2>(
        sys, x, gkoExec_, factory_, stats, l1Control, cachedImap_, cachedNonLocalMtx_
    );
    return stats;
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
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    auto gkoMtx = createGkoMtxDist(
        gkoExec_,
        comm,
        sys.matrix(),
        sys.offDiagonalMatrix(),
        commPattern,
        cachedImap_,
        cachedNonLocalMtx_
    );

    // Implicit transform-BC path: solve the three components segregated, applying each column's
    // per-component diagonal correction to the shared rank-local diagonal in place. Mirrors the
    // serial path; createGkoMtxDist views the local matrix, so the in-place edits are seen at solve
    // time.
    if (sys.diagCmpt() && sys.diagCmpt()->size() > 0)
    {
        auto values = const_cast<Vector<scalar>&>(sys.matrix().values()).view();
        const auto ma = sys.faceToMatrixAddress()->view(sys.matrix().rowOffs().view());
        auto diagC = sys.diagCmpt()->view();
        const localIdx nrows = sys.rhs().size();
        gkoExec_->synchronize();

        SolverStats stats;
        solveImplicitTransformComponentDist<0>(
            sys,
            x,
            exec_,
            gkoExec_,
            comm,
            gkoMtx,
            factory_,
            stats,
            l1Control,
            values,
            ma,
            diagC,
            nrows
        );
        solveImplicitTransformComponentDist<1>(
            sys,
            x,
            exec_,
            gkoExec_,
            comm,
            gkoMtx,
            factory_,
            stats,
            l1Control,
            values,
            ma,
            diagC,
            nrows
        );
        solveImplicitTransformComponentDist<2>(
            sys,
            x,
            exec_,
            gkoExec_,
            comm,
            gkoMtx,
            factory_,
            stats,
            l1Control,
            values,
            ma,
            diagC,
            nrows
        );
        return stats;
    }


    auto solver = gko::share(factory_->generate(gkoMtx));
    return solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, solver, l1Control);
}

template std::
    shared_ptr<const gko::LinOp>
    createGkoMtxDist<
        localIdx>(std::shared_ptr<const gko::Executor>, const gko::experimental::mpi::communicator&, const CSRMatrix<scalar, localIdx>&, const COOMatrix<scalar, localIdx>&, const CommunicationPattern&, std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>&, std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>&);

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
