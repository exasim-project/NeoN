// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO
#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/error.hpp"

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Profiling_ScopedRegion.hpp> // ginkgo.solverSetup sub-region

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
    std::shared_ptr<gko::matrix::Coo<scalar, IndexType>>& nonLocalMtxCache,
    std::shared_ptr<const gko::LinOp>& distMtxCache,
    const scalar*& localValPtrCache,
    const std::string& localMatrixFormat
)
{
    // commPattern is currently unused here: all the connectivity information needed to build
    // the distributed matrix is already encoded in the row/column indices of `mtx` (local block)
    // and `bmtx` (off-diagonal/processor coupling).
    static_cast<void>(commPattern);

    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;

    const auto nNonLocalNnz = static_cast<gko::size_type>(bmtx.values().size());

    // Fast path (steady state): reuse the cached distributed-matrix wrapper when the local CSR value
    // buffer is unchanged (NeoN re-assembles in place). The local block is a non-owning Csr view that
    // already reflects the re-assembled values, and the index_map / partition are fixed, so the whole
    // wrapper is reused -- only the off-diagonal values are refreshed into the cached Coo. This skips
    // the per-solve Csr::create_const, whose default (load_balance) strategy recomputes the srow
    // load-balancing array by scanning the ~O(nnz) sparsity every solve (~72 ms on the 16M-row local
    // block). The pointer guard rebuilds (below) if the value buffer was reallocated. [Restored
    // 2026-07-23: this distributed-matrix cache was dropped in a rebase; its loss was the dominant
    // remaining per-solve host cost -- see cachedDistMtx_ in ginkgo.hpp.]
    if (distMtxCache && nonLocalMtxCache && imapCache && localValPtrCache == mtx.values().data())
    {
        const auto bValV = bmtx.values().view();
        auto* cachedValsPtr = nonLocalMtxCache->get_values();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) { cachedValsPtr[i] = bValV[i]; },
            "updateNonLocalValues"
        );
        return distMtxCache;
    }

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
    const auto matrixSize = gko::dim<2> {nrows, nrows};

    std::shared_ptr<const gko::LinOp> localMtx;
    if (localMatrixFormat == "Sellp")
    {
        auto sellp = gko::share(gko::matrix::Sellp<scalar, IndexType>::create(exec, matrixSize));
        gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
                       exec, matrixSize, std::move(vals), std::move(col), std::move(row)
                   ))
            ->convert_to(sellp);
        localMtx = sellp;
    }
    else
    {
        localMtx = gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
            exec, matrixSize, std::move(vals), std::move(col), std::move(row)
        ));
    }

    if (imapCache && nonLocalMtxCache)
    {
        // Topology is fixed but the wrapper is not cached yet, or the local CSR buffer was
        // reallocated (pointer-guard miss above): refresh the off-diagonal values, (re)build the
        // wrapper around the current local view, and cache it for the values-only fast path.
        const auto bValV = bmtx.values().view();
        auto* cachedValsPtr = nonLocalMtxCache->get_values();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) { cachedValsPtr[i] = bValV[i]; },
            "updateNonLocalValues"
        );
        // [fence-audit] removed redundant fence: this build kernel and the subsequent
        // dist_mtx::create / Ginkgo solve share the Kokkos stream (ginkgo.cpp:200) -> ordered.
        distMtxCache = gko::share(dist_mtx::create(
            exec, comm, *imapCache, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtxCache
        ));
        localValPtrCache = mtx.values().data();
        return distMtxCache;
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
        // [fence-audit] removed redundant fence: buildNonLocalCOO -> Coo/dist_mtx::create, same
        // stream.
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

    distMtxCache = gko::share(dist_mtx::create(
        exec, comm, imap, std::const_pointer_cast<gko::LinOp>(localMtx), nonLocalMtxCache
    ));
    localValPtrCache = mtx.values().data();
    return distMtxCache;
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
        return {
            static_cast<size_t>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration
        };
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

    gko::size_type numIter = logger->get_num_iterations();
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    return {static_cast<size_t>(numIter), initResNorm, finalResNorm, duration};
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
                    {static_cast<size_t>(l1Res.numIter),
                     l1Res.perColInitNorms[i],
                     l1Res.perColFinalNorms[i],
                     duration}
                );
            }
            return stats;
        }
        return {
            static_cast<size_t>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration
        };
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

    gko::size_type numIter = logger->get_num_iterations();
    exec->synchronize();
    auto endEval = std::chrono::steady_clock::now();
    auto duration =
        static_cast<scalar>(
            std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
        )
        / 1000.0;

    SolverStats stats;
    for (int i = 0; i < 3; ++i)
        stats.entries.push_back(
            {static_cast<size_t>(numIter), initNorms[i], finalNorms[i], duration}
        );
    return stats;
}

namespace
{

// Cache-or-update a generated Ginkgo solver across solves (Strategy 1b, see
// docs/plans/ginkgo-solver-reuse-and-shared-allocator.md). On the first solve -- or whenever the
// matrix STRUCTURE changes -- the solver is generated from the factory and cached. On later
// solves the cached solver is reused and only its matrix VALUES are refreshed in place via
// gko::UpdateMatrixValue::update_matrix_value, reusing the (expensive) multigrid Pgm aggregation
// + smoother setup instead of rebuilding the whole hierarchy. The update target is the solver
// itself when it is updatable (Multigrid as the top-level solver) or, for a Krylov solver
// wrapping a Multigrid preconditioner (Cg/Fcg/Ir + MG), the bound preconditioner. The Krylov
// shell's own system matrix needs no explicit refresh: createGkoMtxDist views the local CSR
// zero-copy and refreshes the shared non-local Coo in place, so the cached solver's system matrix
// already tracks the re-assembled values; only the preconditioner's derived Galerkin operators do
// not -- which is exactly what update_matrix_value recomputes. Falls back to a full regenerate
// when no updatable target is found (e.g. a non-multigrid preconditioner).
//
// Caching is opt-in via `cacheEnabled` (the "cacheSolver" dict entry). When disabled, the solver is
// regenerated every solve and the cache state is left untouched. `rebuildInterval` (> 0) forces a
// full regenerate every Nth solve so the preconditioner (Pgm aggregation reused by
// update_matrix_value) is periodically rebuilt from scratch as the matrix values drift across the
// steady iteration; 0 disables the periodic rebuild (update in place forever). `solveCount` tracks
// the number of solves served by the current cached solver and is reset on every (re)generate.
gko::UpdateMatrixValue* findUpdatable(gko::LinOp* solver)
{
    // The solver itself is updatable: Multigrid as the top-level solver.
    if (auto* upd = dynamic_cast<gko::UpdateMatrixValue*>(solver))
    {
        return upd;
    }
    // Krylov solver wrapping an updatable preconditioner: Cg/Fcg + Multigrid, or
    // Cg + Schwarz{Multigrid(local)} (the Schwarz UpdateMatrixValue patch forwards to the
    // per-rank Multigrid).
    if (auto* prec = dynamic_cast<gko::Preconditionable*>(solver))
    {
        if (auto p = prec->get_preconditioner())
        {
            if (auto* upd = dynamic_cast<gko::UpdateMatrixValue*>(const_cast<gko::LinOp*>(p.get())))
            {
                return upd;
            }
        }
    }
    // Ir wrapping an updatable inner SOLVER: scale-corrected MG, Ir(scale_correction){Multigrid}.
    // The Multigrid is Ir's inner solver (get_solver), not its preconditioner; the outer Ir's own
    // system matrix tracks the re-assembled values zero-copy, so refreshing the inner Multigrid
    // (Galerkin coarse ops) completes the reuse.
    if (auto* ir = dynamic_cast<gko::solver::Ir<scalar>*>(solver))
    {
        if (auto s = ir->get_solver())
        {
            if (auto* upd = dynamic_cast<gko::UpdateMatrixValue*>(const_cast<gko::LinOp*>(s.get())))
            {
                return upd;
            }
        }
    }
    return nullptr;
}

// What cacheOrUpdateSolver did this solve, for the p-cache diagnostic.
enum class CacheAction
{
    UpdateInPlace,        // Strategy 1b: refreshed the cached solver's values, no rebuild
    Rebuild,              // generated a fresh solver from scratch (first solve / forced rebuild)
    RebuildReuseWorkspace // Strategy 3: regenerated but reused the stashed scratch Workspace
};

// Handle returned by cacheOrUpdateSolver: the solver to apply, plus -- for the Strategy 3
// workspace-reuse (regenerate) path -- ownership of that solver so its temporary-storage Workspace
// can be reclaimed into the GinkgoSolver's cachedWorkspace_ slot once the solve completes. For the
// Strategy 1b cached path it merely aliases the long-lived cached solver and reclaims nothing.
//
// solver() hands a NON-OWNING alias (no-op deleter) to the solve API so this lease keeps sole
// ownership of a regenerated solver and can still extract its Workspace afterwards. The lease must
// therefore outlive the solve call; it reclaims the Workspace in its destructor.
class SolverLease
{
public:

    // Strategy 1b path: alias a cached / in-place-updated solver. No workspace reclaim.
    SolverLease(std::shared_ptr<gko::LinOp> cached, CacheAction action)
        : solver_(std::move(cached)), action_(action)
    {}

    // Strategy 3 path: own a freshly generated solver and, on destruction, extract its Workspace
    // back into *wsSlot for the next generate() to reuse. `wsSlot` is a GinkgoSolver member and
    // outlives the lease.
    SolverLease(
        std::unique_ptr<gko::LinOp> owned,
        std::unique_ptr<gko::solver::Workspace>* wsSlot,
        CacheAction action
    )
        : owned_(std::move(owned)), wsSlot_(wsSlot), action_(action)
    {
        solver_ = std::shared_ptr<gko::LinOp>(owned_.get(), [](gko::LinOp*) {});
    }

    SolverLease(SolverLease&&) = default;
    SolverLease& operator=(SolverLease&&) = default;
    SolverLease(const SolverLease&) = delete;
    SolverLease& operator=(const SolverLease&) = delete;

    ~SolverLease()
    {
        // wsSlot_ is null for an owned solver whose Workspace must NOT be reused (updatable configs
        // regenerated with caching off -- see cacheOrUpdateSolver): just let owned_ free normally.
        if (owned_ && wsSlot_)
        {
            // Reclaim the scratch Workspace for the next solve. Defensive: a solver that is not
            // workspace-aware throws gko::InvalidStateError -- swallow it (a destructor must not
            // throw) and drop the slot so the next solve seeds a fresh workspace.
            try
            {
                *wsSlot_ = gko::solver::invalidate_and_extract_workspace(std::move(owned_));
            }
            catch (...)
            {
                wsSlot_->reset();
            }
        }
    }

    const std::shared_ptr<gko::LinOp>& solver() const { return solver_; }
    CacheAction action() const { return action_; }

private:

    std::unique_ptr<gko::LinOp> owned_;
    std::unique_ptr<gko::solver::Workspace>* wsSlot_ = nullptr;
    std::shared_ptr<gko::LinOp> solver_;
    CacheAction action_;
};

// Provide a Ginkgo solver for this solve, reusing work across solves. Strategy 1b (updatable
// configs -- Multigrid): cache + update_matrix_value in place. Strategy 3 (non-updatable Krylov --
// PBiCGStab/Cg + Jacobi/ILU): regenerate every solve but reclaim the Krylov scratch Workspace via
// the SolverLease dtor and feed it into the next generate(matrix, ws), amortizing scratch alloc.
// An UPDATABLE config regenerated with caching off must NOT reuse its Workspace (Multigrid coarse
// sizes are value-dependent -> stale reuse aborts the V-cycle with DimensionMismatch).
SolverLease cacheOrUpdateSolver(
    std::shared_ptr<gko::LinOp>& cachedSolver,
    std::unique_ptr<gko::solver::Workspace>& cachedWorkspace,
    std::array<gko::size_type, 3>& cachedStructure,
    localIdx& solveCount,
    bool cacheEnabled,
    localIdx rebuildInterval,
    const std::shared_ptr<const gko::LinOpFactory>& factory,
    const std::shared_ptr<const gko::LinOp>& gkoMtx,
    const std::array<gko::size_type, 3>& structure
)
{
    // Solver setup (generate / update_matrix_value / workspace reuse) runs on every solve and is
    // NOT part of the reported "Solve time". Profiled to expose this unreported solve-path cost.
    Kokkos::Profiling::ScopedRegion region_("ginkgo.solverSetup");

    // Generate a fresh solver, reusing the stashed scratch Workspace when one is available
    // (Strategy 3). The first generate has no workspace yet and uses the solver's own eagerly
    // constructed one, which the SolverLease then extracts to seed cachedWorkspace for reuse.
    bool reusedWorkspace = false;
    auto generateReusing = [&]() -> std::unique_ptr<gko::LinOp>
    {
        if (cachedWorkspace)
        {
            reusedWorkspace = true;
            return factory->generate(gkoMtx, std::move(cachedWorkspace));
        }
        return factory->generate(gkoMtx);
    };

    // Strategy 1b fast path: refresh the cached solver in place when it is updatable and the matrix
    // structure is unchanged and a periodic rebuild is not due.
    if (cacheEnabled)
    {
        const bool periodicRebuild = rebuildInterval > 0 && solveCount >= rebuildInterval;
        if (cachedSolver && cachedStructure == structure && !periodicRebuild)
        {
            if (auto* upd = findUpdatable(cachedSolver.get()))
            {
                upd->update_matrix_value(gkoMtx);
                ++solveCount;
                return SolverLease(cachedSolver, CacheAction::UpdateInPlace);
            }
        }
    }

    // (Re)generate. Decide the tier from the fresh solver's updatability.
    auto fresh = generateReusing();
    cachedStructure = structure;
    solveCount = 1;
    const CacheAction rebuildAction =
        reusedWorkspace ? CacheAction::RebuildReuseWorkspace : CacheAction::Rebuild;

    const bool updatable = findUpdatable(fresh.get()) != nullptr;

    if (cacheEnabled && updatable)
    {
        // Strategy 1b: updatable -> cache the solver and reuse it via update_matrix_value next
        // solve. Its Workspace stays inside the cached solver; cachedWorkspace remains empty.
        cachedSolver = gko::share(std::move(fresh));
        return SolverLease(cachedSolver, rebuildAction);
    }

    cachedSolver = nullptr;
    if (updatable)
    {
        // Updatable config with caching DISABLED (cacheSolver=false): the solver is regenerated
        // from scratch every solve, and its scratch Workspace must NOT be reused (Multigrid coarse
        // vector sizes track the value-dependent Pgm aggregation, which shifts between solves -> a
        // stale Workspace aborts mid-V-cycle with gko::DimensionMismatch). Own the solver for this
        // solve and let its Workspace be freed (wsSlot = nullptr => no reclaim).
        return SolverLease(std::move(fresh), nullptr, rebuildAction);
    }

    // Strategy 3: genuinely non-updatable (fixed-layout Krylov, e.g. PBiCGStab/Cg + Jacobi/ILU) ->
    // the Workspace layout is value-independent across solves, so reclaim it for the next generate.
    return SolverLease(std::move(fresh), &cachedWorkspace, rebuildAction);
}

// Structural key for the solver cache: re-assembling values keeps these fixed (steady SIMPLE), a
// remesh/topology change does not -- on a mismatch the cached solver is dropped and regenerated.
template<typename SystemType>
std::array<gko::size_type, 3> solverStructureKey(const SystemType& sys)
{
    return {
        static_cast<gko::size_type>(sys.matrix().sparsity()->rows()),
        static_cast<gko::size_type>(sys.matrix().values().size()),
        static_cast<gko::size_type>(sys.offDiagonalMatrix().values().size())
    };
}

} // namespace

// ---------------------------------------------------------------------------
// PROTOTYPE (direction #1): fused slip/symmetry momentum operator.
//
// For the implicit slip/symmetry BC the three velocity components share ONE
// scalar CSR matrix but carry a per-component diagonal correction
// diagC[cell][c] (= gamma*|S|*Delta*|n_c|, nonzero only on slip/symmetry
// boundary cells). The legacy path (solveImplicitTransformComponentDist)
// solves the three columns segregated -> three distributed solves -> three
// halo exchanges per Krylov iteration.
//
// This LinOp instead exposes
//     (A_fused * X)[:,c] = A_shared * X[:,c] - diagC[:,c] .* X[:,c]
// as a SINGLE operator over the 3-column distributed multivector, so one
// multi-RHS solve does ONE fused halo exchange for all three components --
// restoring the efficient path the non-slip case already uses (see the coupled
// fallback at the bottom of solveDist).
//
// The diagonal shift is zero-copy: diagCmpt stores Vec3 per cell as contiguous
// [cell][0..2], i.e. the same [cell*3+c] layout as the distributed Vector's
// local block, so diagCmpt->data() is reinterpreted as the per-entry shift with
// no allocation or copy. A_shared (the plain distributed matrix) is reused
// as-is for the SpMV; only the local diagonal contribution is corrected, which
// is a purely rank-local (comm-free) axpy.
class FusedDiagShiftMatrix : public gko::EnableLinOp<FusedDiagShiftMatrix>
{
    friend class gko::EnablePolymorphicObject<FusedDiagShiftMatrix, gko::LinOp>;

public:

    using dist_vec = gko::experimental::distributed::Vector<scalar>;

    static std::shared_ptr<FusedDiagShiftMatrix> create(
        std::shared_ptr<const gko::Executor> gkoExec,
        Executor nfExec,
        std::shared_ptr<const gko::LinOp> baseMtx, // plain distributed A_shared (multi-RHS capable)
        const scalar* shift,                       // [nLocalRows*3], layout [cell*3+c], zero-copy
        localIdx nLocalRows
    )
    {
        return std::shared_ptr<FusedDiagShiftMatrix>(new FusedDiagShiftMatrix(
            std::move(gkoExec), nfExec, std::move(baseMtx), shift, nLocalRows
        ));
    }

    // x[:,c] -= s * diagC[:,c] .* b[:,c] over the local block (diagonal -> no comm).
    // The distributed Vector's local storage is dense-packed row-major [row*cols + col];
    // the momentum solve always carries cols == 3, matching the diagC [cell*3+c] layout.
    // NOTE: must be public -- nvcc forbids an extended __device__ lambda inside a
    // private/protected member function.
    void applyShift(const gko::LinOp* b, gko::LinOp* x, scalar s) const
    {
        const auto* bl = gko::as<const dist_vec>(b)->get_const_local_values();
        auto* xl = gko::as<dist_vec>(x)->get_local_values();
        const auto* lv = gko::as<dist_vec>(x)->get_local_vector();
        const auto cols = static_cast<localIdx>(lv->get_size()[1]);
        const auto rows = static_cast<localIdx>(lv->get_size()[0]);
        const localIdx n = rows * cols;
        const scalar* shift = shift_;
        parallelFor(
            nfExec_,
            {0, n},
            KOKKOS_LAMBDA(const localIdx i) {
                const localIdx row = i / cols;
                const localIdx col = i - row * cols;
                xl[i] -= s * shift[row * 3 + col] * bl[i];
            },
            "fusedDiagShiftApply"
        );
    }

protected:

    // x = A_fused * b  (one fused halo exchange in baseMtx_->apply)
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        baseMtx_->apply(b, x);
        applyShift(b, x, scalar(1.0));
    }

    // x = beta*x + alpha * A_fused * b
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        baseMtx_->apply(alpha, b, beta, x);
        const scalar a = gkoExec_->copy_val_to_host(
            gko::as<const gko::matrix::Dense<scalar>>(alpha)->get_const_values()
        );
        applyShift(b, x, a);
    }

private:

    // never used (no clone()/copy in this path); present only to satisfy the
    // EnablePolymorphicObject default-construction requirement.
    explicit FusedDiagShiftMatrix(std::shared_ptr<const gko::Executor> gkoExec)
        : gko::EnableLinOp<FusedDiagShiftMatrix>(gkoExec), gkoExec_(std::move(gkoExec))
    {}

    FusedDiagShiftMatrix(
        std::shared_ptr<const gko::Executor> gkoExec,
        Executor nfExec,
        std::shared_ptr<const gko::LinOp> baseMtx,
        const scalar* shift,
        localIdx nLocalRows
    )
        : gko::EnableLinOp<FusedDiagShiftMatrix>(gkoExec, baseMtx->get_size()),
          gkoExec_(std::move(gkoExec)), nfExec_(nfExec), baseMtx_(std::move(baseMtx)),
          shift_(shift), nLocalRows_(nLocalRows)
    {}

    std::shared_ptr<const gko::Executor> gkoExec_;
    Executor nfExec_ {SerialExecutor {}};
    std::shared_ptr<const gko::LinOp> baseMtx_;
    const scalar* shift_ = nullptr;
    localIdx nLocalRows_ = 0;
};

template<unsigned int I>
void solveComponentDist(
    auto& sys,
    auto& x,
    auto& exec,
    auto& factory,
    auto& stats,
    const L1ResidualControl* l1Control,
    std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>& imapCache,
    std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>& nonLocalMtxCache,
    const std::string& localMatrixFormat
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
    // Segregated Vec3 component path: not wired to the distributed-matrix cache (each component
    // rebuilds). Pass non-persistent locals so createGkoMtxDist behaves as before (no reuse).
    std::shared_ptr<const gko::LinOp> uncachedDistMtx;
    const scalar* uncachedLocalValPtr = nullptr;
    auto gkoMtx = createGkoMtxDist(
        exec,
        comm,
        mtx,
        nonLocalMtx,
        commPattern,
        imapCache,
        nonLocalMtxCache,
        uncachedDistMtx,
        uncachedLocalValPtr,
        localMatrixFormat
    );
    // NOTE: the segregated Vec3 path cannot reuse the solver cache here: threading a
    // std::shared_ptr<gko::LinOp>& through this device-kernel-launching template (getComponent /
    // setComponent) trips a cudafe++ "__remove_cv(gko::LinOp)" stub bug. Regenerate per solve;
    // caching for this path needs a host-only restructure (see plan TODO).
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
        cachedNonLocalMtx_,
        cachedDistMtx_,
        cachedLocalValPtr_,
        localMatrixFormat_
    );
    auto lease = cacheOrUpdateSolver(
        cachedSolver_[0],
        cachedWorkspace_[0],
        cachedSolverStructure_[0],
        cachedSolveCount_[0],
        cacheSolver_,
        preconditionerRebuildInterval_,
        factory_,
        gkoMtx,
        solverStructureKey(sys)
    );
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, lease.solver(), l1Control)};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    solveComponentDist<0>(
        sys,
        x,
        gkoExec_,
        factory_,
        stats,
        l1Control,
        cachedImap_,
        cachedNonLocalMtx_,
        localMatrixFormat_
    );
    solveComponentDist<1>(
        sys,
        x,
        gkoExec_,
        factory_,
        stats,
        l1Control,
        cachedImap_,
        cachedNonLocalMtx_,
        localMatrixFormat_
    );
    solveComponentDist<2>(
        sys,
        x,
        gkoExec_,
        factory_,
        stats,
        l1Control,
        cachedImap_,
        cachedNonLocalMtx_,
        localMatrixFormat_
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
    // Wired to the distributed-matrix cache. Safe for all three sub-paths below: the wrapper is a
    // non-owning VIEW over the rank-local value buffer, so it always reflects the live values -- it
    // cannot carry stale data. The implicit-transform branch shifts the diagonal in place per
    // component but RESTORES it immediately after each component solve (apply/restore pair below),
    // and the buffer is re-assembled in place each step, so the cached wrapper's value pointer stays
    // valid; the pointer guard in createGkoMtxDist rebuilds if it ever changes. The fused-slip
    // branch applies its shift through FusedDiagShiftMatrix (operator-level, no buffer mutation).
    auto gkoMtx = createGkoMtxDist(
        gkoExec_,
        comm,
        sys.matrix(),
        sys.offDiagonalMatrix(),
        commPattern,
        cachedImap_,
        cachedNonLocalMtx_,
        cachedDistMtx_,
        cachedLocalValPtr_,
        localMatrixFormat_
    );

    // Implicit transform-BC path: solve the three components segregated, applying each column's
    // per-component diagonal correction to the shared rank-local diagonal in place. Mirrors the
    // serial path; createGkoMtxDist views the local matrix, so the in-place edits are seen at solve
    // time.
    if (sys.diagCmpt() && sys.diagCmpt()->size() > 0)
    {
        // PROTOTYPE (direction #1): fuse the three components into a single multi-RHS solve with a
        // per-column diagonal shift (FusedDiagShiftMatrix above), so one solve does ONE fused halo
        // exchange instead of three. Gated on NEON_FUSED_SLIP_SOLVE (default ON; set to 0 to fall
        // back to the legacy segregated path) and on the L1 stop being active -- the L1 criterion
        // overrides the solver's own stopping factory and measures the true fused residual (it is
        // built with_matrix(fusedOp)), so the hand-built Bicgstab below needs no faithful criteria.
        static const bool fusedSlip = []
        {
            const char* e = std::getenv("NEON_FUSED_SLIP_SOLVE");
            return !(e != nullptr && std::string(e) == "0");
        }();
        if (fusedSlip && l1Control != nullptr)
        {
            gkoExec_->synchronize();

            // Zero-copy per-entry diagonal shift: diagCmpt's Vec3 storage [cell][0..2] is
            // contiguous, matching the distributed multivector's local [cell*3+c] layout. The sign
            // matches the legacy path's atomic_sub(diagC): A_fused = A_shared - diag(diagC) per
            // column.
            const scalar* shift = reinterpret_cast<const scalar*>(sys.diagCmpt()->data());
            const localIdx nrows = sys.rhs().size();
            auto fusedOp = FusedDiagShiftMatrix::create(gkoExec_, exec_, gkoMtx, shift, nrows);

            // Borrow the config's preconditioner (Schwarz{Jacobi} for the distributed diagonal
            // preconditioner) by generating a reference solver on the PLAIN distributed matrix --
            // Schwarz needs a real distributed::Matrix (get_local_matrix) that fusedOp is not. The
            // preconditioner approximates A_shared, ignoring the boundary-only diagC shift; that
            // only affects convergence rate, not correctness (the fused operator carries the exact
            // system).
            std::shared_ptr<const gko::LinOp> precond;
            if (auto p = std::dynamic_pointer_cast<const gko::Preconditionable>(
                    gko::share(factory_->generate(gkoMtx))
                ))
            {
                precond = p->get_preconditioner();
            }

            // Single fused multi-RHS solve. solve_impl_dist handles the 3-column dist_vec,
            // per-column L1 norms and stats; passing fusedOp as the residual operator keeps
            // everything exact.
            auto solver =
                gko::share(gko::solver::Bicgstab<scalar>::build()
                               .with_generated_preconditioner(precond)
                               .with_criteria(gko::stop::Iteration::build().with_max_iters(10000u))
                               .on(gkoExec_)
                               ->generate(fusedOp));

            return solve_impl_dist(gkoExec_, comm, sys.rhs(), x, fusedOp, solver, l1Control);
        }

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


    // Coupled fallback (no implicit transform BC): one block solve over the Vec3 rhs / scalar
    // matrix. Cache on slot [0] -- the segregated path is not taken for this system, so slots
    // [1..2] stay free.
    auto lease = cacheOrUpdateSolver(
        cachedSolver_[0],
        cachedWorkspace_[0],
        cachedSolverStructure_[0],
        cachedSolveCount_[0],
        cacheSolver_,
        preconditionerRebuildInterval_,
        factory_,
        gkoMtx,
        solverStructureKey(sys)
    );
    return solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, lease.solver(), l1Control);
}

template std::
    shared_ptr<const gko::LinOp>
    createGkoMtxDist<
        localIdx>(std::shared_ptr<const gko::Executor>, const gko::experimental::mpi::communicator&, const CSRMatrix<scalar, localIdx>&, const COOMatrix<scalar, localIdx>&, const CommunicationPattern&, std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>&, std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>&, std::shared_ptr<const gko::LinOp>&, const scalar*&, const std::string&);

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
