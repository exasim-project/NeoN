// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO
#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/distributed/communicationPattern.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/error.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <unordered_map>
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

// --- Ginkgo distributed-matrix skeleton registry (D-01 / D-05 / D-06) ---

using dist_mtx_t = gko::experimental::distributed::Matrix<scalar, label, gko::int64>;
using local_csr_t = gko::matrix::Csr<scalar, label>;
using nonlocal_coo_t = gko::matrix::Coo<scalar, label>;

struct GinkgoSkeletonEntry
{
    std::shared_ptr<dist_mtx_t> distMtx;
    std::shared_ptr<local_csr_t> mutableLocalCsr;       // non-const; OWNED value buffer
    std::shared_ptr<nonlocal_coo_t> mutableNonLocalCoo; // non-const; owns nlRow/nlCol/nlVal arrays
    // Non-local COO data is owned by mutableNonLocalCoo (Ginkgo-managed, no Kokkos alloc).
    // Static-registry safety: Ginkgo uses cudaFree/free directly, not Kokkos, so these survive
    // Kokkos finalization without a crash during __cxa_finalize.
    std::vector<label> rowSortPerm; // host-side row-sort permutation (for NF_ASSERT in hit path)
    std::size_t buildCount {0};     // D-06 always-on skeleton-build counter
};

// Thread-safety: solves are sequential per rank; the registry is touched only from
// createGkoMtxDist (via solveDist/solveComponentDist). No locking required.
static std::unordered_map<const void*, GinkgoSkeletonEntry> gSkeletonRegistry;

// PROFILING TOGGLE (temporary, env-gated): NEON_GKO_NOCACHE=1 bypasses the skeleton
// cache entirely — every solve rebuilds a fresh dist_mtx (no registry lookup/store),
// mirroring develop's per-solve build. Lets one binary A/B the cache cost.
static const bool gNoCache = (std::getenv("NEON_GKO_NOCACHE") != nullptr);

// Registration flag: the Kokkos finalize hook is registered the first time createGkoMtxDist
// is called (Kokkos is guaranteed to be initialized at that point). The hook clears the
// registry during Kokkos::finalize(), which fires before __cxa_finalize (static destructors),
// ensuring Ginkgo CUDA objects are freed while the CUDA driver is still active.
// Without this hook, the static destructor for gSkeletonRegistry fires after CUDA unloads,
// causing cudaErrorCudartUnloading in gko::cuda_scoped_device_id_guard.
static bool gKokkosHookRegistered = false;

// D-06: always-on test accessor — declared in ginkgo.hpp under NF_WITH_MPI_SUPPORT.
// Returns the per-key skeleton build count, or 0 if the key has never been seen.
std::size_t getSkeletonBuildCount(const void* key)
{
    auto it = gSkeletonRegistry.find(key);
    return (it != gSkeletonRegistry.end()) ? it->second.buildCount : 0;
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
    // Register the Kokkos finalize hook once (Kokkos is guaranteed initialized at this call site).
    // The hook clears gSkeletonRegistry before CUDA shuts down, avoiding cudaErrorCudartUnloading
    // in gko::cuda_scoped_device_id_guard during __cxa_finalize.
    if (!gKokkosHookRegistered)
    {
        Kokkos::push_finalize_hook([]() { gSkeletonRegistry.clear(); });
        gKokkosHookRegistered = true;
    }

    // D-03: key on the stable SparsityPattern identity (stable because D-04 caches it in stencilDB
    // via cachedSparsityPattern; a new mesh yields distinct SparsityPattern objects → cache miss).
    const void* key = static_cast<const void*>(mtx.sparsity().get());

    auto it = gNoCache ? gSkeletonRegistry.end() : gSkeletonRegistry.find(key);
    if (it != gSkeletonRegistry.end())
    {
        // CACHE HIT — refresh matrix values only (D-05). No collectives, no dist_mtx::create.
        auto& entry = it->second;

        // Refresh local CSR values into the owned buffer. The sparsity (colIdxs/rowOffs) is
        // structural and mesh-invariant (stable after D-04); only the numerical values change.
        const auto nnz = static_cast<localIdx>(mtx.values().size());
        {
            auto srcView = mtx.values().view();
            auto* dstPtr = entry.mutableLocalCsr->get_values();
            parallelFor(
                mtx.exec(),
                {0, nnz},
                KOKKOS_LAMBDA(const localIdx i) { dstPtr[i] = srcView[i]; },
                "refreshLocalCsrValues"
            );
        }

        // Refresh non-local COO values. bmtx.values() is stored in row-sorted order (operators
        // write via BoundaryMesh::getRowOrderWriteIndex() = invPerm, i.e. sorted position).
        // offDiagRowSortPerm[i] = assembly-order index for sorted position i; bmtx.values()[i]
        // is already at sorted position i, so a direct copy preserves the CUDA Coo::apply2
        // row-sort invariant (project_gpu_distributed_pcg_negdef_diverges).
        const auto nNlNnz = static_cast<localIdx>(bmtx.values().size());
        {
            const auto& perm = commPattern.offDiagRowSortPerm;
            NF_ASSERT(
                perm.empty() || perm.size() == static_cast<std::size_t>(nNlNnz),
                "offDiagRowSortPerm size mismatch on COO value refresh"
            );
            auto bmtxValView = bmtx.values().view();
            auto* dstPtr = entry.mutableNonLocalCoo->get_values();
            parallelFor(
                bmtx.exec(),
                {0, nNlNnz},
                KOKKOS_LAMBDA(const localIdx i) { dstPtr[i] = bmtxValView[i]; },
                "refreshNonLocalCooValues"
            );
        }
        fence(bmtx.exec());
        return entry.distMtx;
    }

    // --- CACHE MISS: build the skeleton once (NeighborhoodCommunicator runs here only) ---

    using global_index_type = gko::int64;
    using dist_mtx = gko::experimental::distributed::Matrix<scalar, label, global_index_type>;

    const auto nrows = static_cast<gko::size_type>(mtx.sparsity()->rows());

    // build_partition_from_local_size fires MPI_Allgather — MISS path only (SOLVER-01).
    auto partition = gko::share(
        gko::experimental::distributed::build_partition_from_local_size<label, global_index_type>(
            exec, comm, nrows
        )
    );

    // Off-diagonal block: rowIdxs()/colIdxs() are pre-sorted by ascending faceOwner (local row)
    // from the assembly phase — no host copies or sort are needed here.
    const auto nNonLocalNnz = static_cast<gko::size_type>(bmtx.values().size());

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

    auto imap = gko::experimental::distributed::index_map<label, global_index_type>(
        exec, partition, comm.rank(), recv_connections
    );
    const auto numNonLocalElements = imap.get_non_local_size();

    // Map global column indices into the non-local index space. Every off-diagonal entry maps to
    // a known remote column — the assembly phase guarantees this by construction.
    const auto mapped =
        imap.map_to_local(recv_connections, gko::experimental::distributed::index_space::non_local);

    // Build non-local COO arrays (nlRow/nlCol/nlVal) via NeoN parallelFor, then own the data in
    // Ginkgo-managed arrays. Ginkgo arrays use Ginkgo's executor allocator (cudaFree/free directly)
    // so they safely outlive Kokkos finalization (avoids static-destructor crash on program exit).
    auto gkoNlVal = gko::array<scalar>(exec, nNonLocalNnz);
    auto gkoNlRow = gko::array<label>(exec, nNonLocalNnz);
    auto gkoNlCol = gko::array<label>(exec, nNonLocalNnz);
    {
        auto* nlRowPtr = gkoNlRow.get_data();
        auto* nlColPtr = gkoNlCol.get_data();
        auto* nlValPtr = gkoNlVal.get_data();
        const auto bRowV = bmtx.sparsity()->rowIdxs().view();
        const auto bValV = bmtx.values().view();
        const auto* mappedPtr = mapped.get_const_data();
        parallelFor(
            bmtx.exec(),
            {0, static_cast<localIdx>(nNonLocalNnz)},
            KOKKOS_LAMBDA(const localIdx i) {
                nlRowPtr[i] = static_cast<label>(bRowV[i]);
                nlColPtr[i] = static_cast<label>(mappedPtr[i]);
                nlValPtr[i] = bValV[i];
            },
            "buildNonLocalCOO"
        );
        fence(bmtx.exec());
    }

    // Host-side row-sort permutation — used in the cache-hit path to assert size contract.
    const auto& hostPerm = commPattern.offDiagRowSortPerm;
    std::vector<label> rowSortPerm(hostPerm.begin(), hostPerm.end());

    // Local CSR: OWNED value buffer (not const_view — values must be mutable for per-solve
    // refresh). Structure arrays (colIdxs/rowOffs) use stable views from the mesh-cached
    // SparsityPattern (D-04).
    const auto nnzLocal = static_cast<gko::size_type>(mtx.values().size());
    auto ownedVals = gko::array<scalar>(exec, nnzLocal);
    {
        auto srcView = mtx.values().view();
        auto* dstPtr = ownedVals.get_data();
        parallelFor(
            mtx.exec(),
            {0, static_cast<localIdx>(nnzLocal)},
            KOKKOS_LAMBDA(const localIdx i) { dstPtr[i] = srcView[i]; },
            "initLocalCsrValues"
        );
        fence(mtx.exec());
    }
    // Structure: stable pointers from mesh-cached SparsityPattern (D-04); const_cast safe because
    // the array views are non-owning and the SparsityPattern lifetime exceeds the Csr lifetime.
    auto colView = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->colIdxs().size()),
        const_cast<IndexType*>(mtx.sparsity()->colIdxs().data())
    );
    auto rowView = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.sparsity()->rowOffs().size()),
        const_cast<IndexType*>(mtx.sparsity()->rowOffs().data())
    );
    auto mutableLocalCsr = gko::share(local_csr_t::create(
        exec,
        gko::dim<2> {nrows, nrows},
        std::move(ownedVals),
        std::move(colView),
        std::move(rowView)
    ));

    // Non-local COO: Coo takes ownership of the Ginkgo arrays (OWNING, not view).
    // After std::move into the Coo, the entry holds the Coo via mutableNonLocalCoo;
    // entry.nlVal/nlRow/nlCol are now EMPTY (moved-from) — but Coo owns the data.
    // The hit path calls mutableNonLocalCoo->get_values() to refresh values in-place.
    auto mutableNonLocalCoo = gko::share(nonlocal_coo_t::create(
        exec,
        gko::dim<2> {nrows, numNonLocalElements},
        std::move(gkoNlVal),
        std::move(gkoNlCol),
        std::move(gkoNlRow)
    ));

    // dist_mtx::create builds the NeighborhoodCommunicator (MPI_Alltoall +
    // MPI_Dist_graph_create_adjacent) — MISS path only (SOLVER-01).
    auto cachedDist = gko::share(
        dist_mtx::create(exec, comm, std::move(imap), mutableLocalCsr, mutableNonLocalCoo)
    );

    // NEON_GKO_NOCACHE: skip the registry store — return the fresh matrix (rebuilt next solve).
    if (gNoCache) return cachedDist;

    // Store entry: mutableLocalCsr/mutableNonLocalCoo retained for per-solve value refresh.
    // nlRow/nlCol/nlVal data is owned by mutableNonLocalCoo (Ginkgo-managed, Coo::create owns
    // them).
    auto [insIt, ok] = gSkeletonRegistry.emplace(
        key,
        GinkgoSkeletonEntry {
            cachedDist, mutableLocalCsr, mutableNonLocalCoo, std::move(rowSortPerm), 1
        }
    );
    (void)ok;
    return insIt->second.distMtx;
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
    auto solver = gko::share(factory->generate(gkoMtx));
    stats.entries.push_back(solve_impl_dist(exec, comm, rhs, xcopy, gkoMtx, solver, l1Control));
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
    solveComponentDist<0>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<1>(sys, x, gkoExec_, factory_, stats, l1Control);
    solveComponentDist<2>(sys, x, gkoExec_, factory_, stats, l1Control);
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
    auto gkoMtx =
        createGkoMtxDist(gkoExec_, comm, sys.matrix(), sys.offDiagonalMatrix(), commPattern);
    auto solver = gko::share(factory_->generate(gkoMtx));
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    return solve_impl_dist(gkoExec_, comm, sys.rhs(), x, gkoMtx, solver, l1Control);
}

template std::shared_ptr<const gko::LinOp> createGkoMtxDist<
    localIdx>(std::shared_ptr<const gko::Executor>, const gko::experimental::mpi::communicator&, const CSRMatrix<scalar, localIdx>&, const COOMatrix<scalar, localIdx>&, const CommunicationPattern&);

}

#endif // NF_WITH_MPI_SUPPORT
#endif // NF_WITH_GINKGO
