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
#include <iostream>
#include <memory>
#include <vector>

#include <Kokkos_Profiling_ScopedRegion.hpp> // profiling sub-regions (no-op without a kokkos tool)

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
    const std::string& localMatrixFormat
)
{
    // Building the distributed Ginkgo matrix runs on EVERY solve and is NOT part of the reported
    // "Solve time" (solve_impl_dist times only the apply): on the cached fast path it refreshes the
    // non-local COO values + re-wraps the local CSR view; on the first solve it also builds the
    // partition / index_map / column mapping. Profiled to expose this unreported solve-path cost.
    Kokkos::Profiling::ScopedRegion region_("ginkgo.createMtx");

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

namespace
{

// Return the gko::UpdateMatrixValue facet reachable from `solver` -- the solver itself, its bound
// preconditioner, or an Ir's inner solver -- or nullptr if none. This is exactly the set of configs
// Strategy 1b can refresh in place (Multigrid as the top-level solver; a Krylov solver wrapping a
// Multigrid / Schwarz{Multigrid} preconditioner; an Ir wrapping a Multigrid). Raw-pointer form so
// it works on both the cached shared_ptr and a freshly generated unique_ptr.
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
        if (owned_)
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

// Provide a Ginkgo solver for this solve, reusing work across solves where possible
// (docs/plans/ginkgo-solver-reuse-and-shared-allocator.md):
//
//   Strategy 1b (updatable configs -- Multigrid as solver/preconditioner, incl. Schwarz{MG} and
//     Ir{MG}): cache the generated solver and, on later solves with unchanged structure, refresh
//     its matrix VALUES in place via gko::UpdateMatrixValue::update_matrix_value -- reusing the
//     expensive Pgm aggregation + smoother setup instead of rebuilding the hierarchy. The Krylov
//     shell's own system matrix needs no explicit refresh (createGkoMtxDist views the local CSR
//     zero-copy + refreshes the non-local Coo in place); only the preconditioner's derived Galerkin
//     operators do, which is what update_matrix_value recomputes.
//
//   Strategy 3 (NON-updatable configs -- e.g. PBiCGStab/Cg/Fcg + Jacobi/ILU/no preconditioner):
//     update_matrix_value does not apply, so the solver must be regenerated every solve. Rather
//     than re-allocate the Krylov scratch vectors each time, the solver's temporary-storage
//     Workspace is extracted after each solve (SolverLease dtor) and fed back into the next
//     generate(matrix, workspace), amortizing the scratch allocation. Such solvers are NOT cached
//     (caching a solver that is rebuilt every solve buys nothing); only their Workspace persists.
//
// `cacheEnabled` (the "cacheSolver" dict entry) gates Strategy 1b; Strategy 3 always applies on a
// regenerate (it is purely an allocation optimization and never changes results). `rebuildInterval`
// (> 0) forces a full Strategy 1b rebuild every Nth solve so the reused Pgm aggregation can't drift
// unboundedly; 0 updates in place forever. `solveCount` tracks solves served by the current cached
// solver, reset on every (re)generate.
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

    if (cacheEnabled && findUpdatable(fresh.get()))
    {
        // Strategy 1b: updatable -> cache the solver and reuse it via update_matrix_value next
        // solve. Its Workspace stays inside the cached solver; cachedWorkspace remains empty.
        cachedSolver = gko::share(std::move(fresh));
        return SolverLease(cachedSolver, rebuildAction);
    }

    // Strategy 3: non-updatable (or caching disabled) -> do not cache the solver (it would be
    // regenerated every solve anyway); reclaim its Workspace for the next generate instead.
    cachedSolver = nullptr;
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

template<unsigned int I>
void solveComponentDist(
    auto& sys,
    auto& x,
    auto& exec,
    auto& factory,
    auto& stats,
    const L1ResidualControl* l1Control,
    auto& scratch, // GinkgoSolver::ComponentScratch& -- persistent per-component buffers (#2a)
    std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>& imapCache,
    std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>& nonLocalMtxCache,
    const std::string& localMatrixFormat
)
{
    // Profiling sub-regions to localise the momentumPredictor cost: matrixPrep (component extract +
    // distributed-matrix build), generate (per-component BiCGStab/Jacobi factory->generate), apply
    // (the actual solve). No-op unless a kokkos-tools connector is loaded.
    Kokkos::Profiling::pushRegion("momentum.matrixPrep");
    auto sparsity = sys.matrix().sparsity();
    auto nonLocalSparsity = sys.offDiagonalMatrix().sparsity();
    const auto srcExec = sys.matrix().values().exec();

    // #2a: refresh the persistent per-component scalar matrices / vectors IN PLACE instead of
    // allocating fresh ones every solve. The scalar CSR/COO matrices are constructed ONCE (lazily,
    // with the fixed sparsity); thereafter only their component values are overwritten via the
    // in-place getComponent (size-guarded -> no realloc in steady state). createGkoMtxDist then
    // VIEWS scratch.csr's values zero-copy, so there is no per-solve big allocation for the
    // momentum predictor (previously the dominant momentumPredictor host-allocation churn).
    if (!scratch.csr)
    {
        scratch.csr.emplace(Vector<scalar>(srcExec, sys.matrix().values().size()), sparsity);
    }
    getComponent<I>(sys.matrix().values(), scratch.csr->values());

    if (!scratch.coo)
    {
        scratch.coo.emplace(
            Vector<scalar>(srcExec, sys.offDiagonalMatrix().values().size()), nonLocalSparsity
        );
    }
    getComponent<I>(sys.offDiagonalMatrix().values(), scratch.coo->values());

    getComponent<I>(sys.rhs(), scratch.rhs);
    getComponent<I>(x, scratch.x);

    const CommunicationPattern& commPattern = sys.commPattern();
    auto comm = gko::experimental::mpi::communicator(
        commPattern.env.comm(), !commPattern.env.gpuAwareMpi()
    );
    auto gkoMtx = createGkoMtxDist(
        exec,
        comm,
        *scratch.csr,
        *scratch.coo,
        commPattern,
        imapCache,
        nonLocalMtxCache,
        localMatrixFormat
    );
    Kokkos::Profiling::popRegion(); // momentum.matrixPrep

    // NOTE: the segregated Vec3 path still regenerates the solver each solve -- the gko::LinOp
    // cache cannot be threaded through this device-kernel-launching template (cudafe++ stub bug,
    // confirmed unfixed on CUDA 13.1.1). #2a only removes the per-solve matrix/vector ALLOCATION;
    // solver-generation reuse (Layer B) still needs the host-only restructure.
    std::shared_ptr<gko::LinOp> solver;
    {
        Kokkos::Profiling::ScopedRegion gen {"momentum.generate"};
        solver = gko::share(factory->generate(gkoMtx));
    }
    {
        Kokkos::Profiling::ScopedRegion app {"momentum.apply"};
        stats.entries.push_back(
            solve_impl_dist(exec, comm, scratch.rhs, scratch.x, gkoMtx, solver, l1Control)
        );
    }
    setComponent<I>(scratch.x, x);
}

// Apply (sign=-1) or restore (sign=+1) component I's implicit-transform diagonal correction to the
// shared rank-local diagonal in place. Split out from the solve so the gko::LinOp solver cache can
// live in the host-only solveDist member: a `std::shared_ptr<gko::LinOp>&` in the signature of a
// template that LEXICALLY contains a __device__ NEON_LAMBDA makes cudafe++ emit a bogus
// `__remove_cv(gko::LinOp)` stub and fail to compile (see neon-cudafe-gko-linop-signature). This
// helper takes only field views (no gko types), so it is safe; the caller does the caching.
//
// NOTE: explicit template parameters (not abbreviated `auto` params): nvcc rejects the extended
// __device__ NEON_LAMBDA parallelFor body when it sits inside an abbreviated function template.
template<
    unsigned int I,
    typename ExecType,
    typename ValuesType,
    typename MatrixAddressingType,
    typename DiagCType>
void applyTransformDiagDist(
    const ExecType& exec,
    ValuesType values,
    const MatrixAddressingType& ma,
    DiagCType diagC,
    localIdx nrows,
    scalar sign
)
{
    Kokkos::Profiling::ScopedRegion region_("ginkgo.transformDiag");
    parallelFor(
        exec,
        {0, nrows},
        NEON_LAMBDA(const localIdx cell) {
            Kokkos::atomic_add(&values[ma.diagIdx(cell)], sign * diagC[cell][I]);
        },
        "applyTransformDiagDist"
    );
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
    // When solver caching is on, report how this solve was served so a run log proves the reuse
    // path is exercised: update_matrix_value reuse (Strategy 1b), a from-scratch rebuild, or a
    // regenerate that reused the cached scratch workspace (Strategy 3 -- the non-updatable path,
    // e.g. PBiCGStab). Rank 0 only; opt-in feature.
    if (cacheSolver_ && comm.rank() == 0)
    {
        const char* action = "rebuild(generate)";
        switch (lease.action())
        {
        case CacheAction::UpdateInPlace:
            action = "reuse(update_matrix_value)";
            break;
        case CacheAction::RebuildReuseWorkspace:
            action = "rebuild(generate)+reuse(workspace)";
            break;
        case CacheAction::Rebuild:
            action = "rebuild(generate)";
            break;
        }
        std::cout << "[GinkgoSolver] p-cache: " << action << " solve=" << cachedSolveCount_[0]
                  << " rebuildInterval=" << preconditionerRebuildInterval_ << std::endl;
    }
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    // `lease` outlives this solve: its destructor reclaims the scratch workspace (Strategy 3) after
    // solve_impl_dist returns.
    return {solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, lease.solver(), l1Control)};
}

SolverStats GinkgoSolver::solveDist(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    // Lazily construct + return the persistent per-component scratch (#2a). exec_ is the NeoN
    // executor the system lives on; the scalar CSR/COO matrices inside are emplaced on first use.
    auto scratch = [this](unsigned int i) -> ComponentScratch&
    {
        if (!cmptScratch_[i]) cmptScratch_[i].emplace(exec_);
        return *cmptScratch_[i];
    };
    solveComponentDist<0>(
        sys,
        x,
        gkoExec_,
        factory_,
        stats,
        l1Control,
        scratch(0),
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
        scratch(1),
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
        scratch(2),
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
    auto gkoMtx = createGkoMtxDist(
        gkoExec_,
        comm,
        sys.matrix(),
        sys.offDiagonalMatrix(),
        commPattern,
        cachedImap_,
        cachedNonLocalMtx_,
        localMatrixFormat_
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
        // No gkoExec_->synchronize() here: getGkoExecutor threads the Kokkos execution-space stream
        // into the Ginkgo executor, so the diagonal-edit kernels below and the Ginkgo solve run on
        // the SAME CUDA stream and are already ordered -- the explicit fences were redundant
        // host-blocking points (EXPERIMENT: removed to measure the slip/symmetry momentum cost).

        SolverStats stats;
        // Same matrix structure for all three components (only the diagonal values differ), so one
        // structure key drives the per-slot cache decision (update-in-place vs rebuild).
        const auto structureKey = solverStructureKey(sys);

        // Solve each component segregated, reusing its OWN cached solver (slot I) across timesteps
        // instead of regenerating it every solve -- the 3x-per-step solver+preconditioner rebuild
        // dominated the slip/symmetry momentum predictor. applyTransformDiagDist edits the shared
        // rank-local diagonal in place (createGkoMtxDist views it), so cacheOrUpdateSolver
        // (update_matrix_value / generate) and the matvec both see matrix_I = base -
        // diag(diagC[:,I]).
        //
        // Inlined here (not a templated helper): a `std::shared_ptr<gko::LinOp>&` in a TEMPLATE
        // function signature makes cudafe++ emit a bogus `__remove_cv(gko::LinOp)` and fail (see
        // neon-cudafe-gko-linop-signature). This member is non-template, like the block path below,
        // so the cache refs are fine here; only the compile-time-I view kernels are templated and
        // they carry no gko types. The three blocks are unrolled because I must be a constant.
#define NEON_SOLVE_TRANSFORM_CMPT(I)                                                               \
    applyTransformDiagDist<I>(exec_, values, ma, diagC, nrows, scalar(-1));                        \
    {                                                                                              \
        auto lease = cacheOrUpdateSolver(                                                          \
            cachedSolver_[I],                                                                      \
            cachedWorkspace_[I],                                                                   \
            cachedSolverStructure_[I],                                                             \
            cachedSolveCount_[I],                                                                  \
            cacheSolver_,                                                                          \
            preconditionerRebuildInterval_,                                                        \
            factory_,                                                                              \
            gkoMtx,                                                                                \
            structureKey                                                                           \
        );                                                                                         \
        auto rhs = getComponent<I>(sys.rhs());                                                     \
        auto xcopy = getComponent<I>(x);                                                           \
        stats.entries.push_back(                                                                   \
            solve_impl_dist(gkoExec_, comm, rhs, xcopy, gkoMtx, lease.solver(), l1Control)         \
        );                                                                                         \
        setComponent<I>(xcopy, x);                                                                 \
    }                                                                                              \
    applyTransformDiagDist<I>(exec_, values, ma, diagC, nrows, scalar(1));

        NEON_SOLVE_TRANSFORM_CMPT(0)
        NEON_SOLVE_TRANSFORM_CMPT(1)
        NEON_SOLVE_TRANSFORM_CMPT(2)
#undef NEON_SOLVE_TRANSFORM_CMPT
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
    // `lease` outlives this solve: its destructor reclaims the scratch workspace (Strategy 3).
    return solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, lease.solver(), l1Control);
}

template std::
    shared_ptr<const gko::LinOp>
    createGkoMtxDist<
        localIdx>(std::shared_ptr<const gko::Executor>, const gko::experimental::mpi::communicator&, const CSRMatrix<scalar, localIdx>&, const COOMatrix<scalar, localIdx>&, const CommunicationPattern&, std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>&, std::shared_ptr<gko::matrix::Coo<scalar, localIdx>>&, const std::string&);

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
