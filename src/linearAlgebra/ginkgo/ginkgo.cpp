// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <array>
#include <map>
#include <mutex>
#include <sstream>

#include <Kokkos_Core.hpp> // Kokkos::Profiling::{push,pop}Region

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

#ifdef NEON_GINKGO_PUBLIC_WORKSPACE
// Public-workspace API from Ginkgo PR #2000 (only present when pinned to that PR via
// NeoN_GINKGO_PUBLIC_WORKSPACE; see cmake/Versions.cmake). The umbrella header should pull it in,
// but include it explicitly so the reuse path fails loudly if the pin is wrong.
#include <ginkgo/core/solver/workspace.hpp>
#endif

gko::config::pnode NeoN::la::ginkgo::parse(const Dictionary& dictIn)
{
    Dictionary dict = dictIn;
    // remove 'solver Ginkgo;' entry
    if (dict.contains("solver") && std::any_cast<std::string>(dict["solver"]) == "Ginkgo")
    {
        dict.remove("solver");
    }

    if (dict.contains("coupled"))
    {
        dict.remove("coupled");
    }

    // 'reportName' is a human-readable solver label (e.g. DICPCG) carried for
    // residual reporting; it is not a Ginkgo config key.
    if (dict.contains("reportName"))
    {
        dict.remove("reportName");
    }

    // 'l1ScaledResidual' opts into the L1-scaled residual stopping criterion
    // (handled separately via readL1ResidualControl); it is not a Ginkgo config key.
    if (dict.contains("l1ScaledResidual"))
    {
        dict.remove("l1ScaledResidual");
    }

    // 'reuseKey' is the per-field solver-factory cache key (consumed by the GinkgoSolver ctor
    // before parse); it is not a Ginkgo config key.
    if (dict.contains("reuseKey"))
    {
        dict.remove("reuseKey");
    }

    // 'checkFrequency' controls how often the L1 criterion evaluates the true residual
    // (handled via readL1ResidualControl); it is not a Ginkgo config key.
    if (dict.contains("checkFrequency"))
    {
        dict.remove("checkFrequency");
    }

    // 'minIter' / 'minIterFactor' steer the L1 criterion's minimum iteration count
    // (handled via readL1ResidualControl and the NeoFOAM PDESolver minIter steering);
    // they are not Ginkgo config keys.
    for (const auto& key : {std::string("minIter"), std::string("minIterFactor")})
    {
        if (dict.contains(key))
        {
            dict.remove(key);
        }
    }

    // check if an external file name is given
    if (dict.contains("configFile"))
    {
        std::string fn_str {};
        auto fn = dict["configFile"];
        if (fn.type() == typeid(std::string))
        {
            fn_str = std::any_cast<std::string>(fn);
        }
        else
        {
            auto token = std::any_cast<TokenList>(fn);
            std::stringstream s;
            for (NeoN::size_t i = 0; i < token.size() - 1; i++)
            {
                s << token.next<std::string>() << "/";
            }
            s << token.next<std::string>();
            fn_str = s.str();
        }

        return gko::ext::config::parse_json_file(fn_str);
    }

    auto parseData = [&](auto key)
    {
        auto parseAny = [&](auto blueprint)
        {
            using value_type = decltype(blueprint);
            if (dict[key].type() == typeid(value_type))
            {
                if constexpr (std::is_same_v<value_type, float>)
                    return gko::config::pnode(static_cast<double>(dict.get<value_type>(key)));
                else
                    return gko::config::pnode(dict.get<value_type>(key));
            }
            else
            {
                return gko::config::pnode();
            }
        };

        if (auto node = parseAny(std::string()))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<const char*>(nullptr)))
        {
            return node;
        }
        if (auto node = parseAny(int {}))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<unsigned int>(0)))
        {
            return node;
        }
        if (auto node = parseAny(double {}))
        {
            return node;
        }
        if (auto node = parseAny(float {}))
        {
            return node;
        }

        NF_THROW("Dictionary key " + key + " has unsupported type: " + dict[key].type().name());
    };
    gko::config::pnode::map_type result;
    for (const auto& key : dict.keys())
    {
        gko::config::pnode node;
        if (dict.isDict(key))
        {
            node = parse(dict.subDict(key));
        }
        else
        {
            node = parseData(key);
        }
        result.emplace(key, node);
    }
    return gko::config::pnode {result};
}


namespace
{

// One memoized Ginkgo executor per NeoN executor kind (variant index: 0 Serial, 1 CPU, 2 GPU).
std::array<std::shared_ptr<gko::Executor>, 3>& gkoExecutorCache()
{
    static std::array<std::shared_ptr<gko::Executor>, 3> cache;
    return cache;
}

std::shared_ptr<gko::Executor> createGkoExecutor(NeoN::Executor exec)
{
    // Defer to Ginkgo's Kokkos extension instead of a hand-rolled #ifdef mapping. Each NeoN
    // executor exposes its backing Kokkos execution space as ExecType::exec (Kokkos::Serial,
    // DefaultHostExecutionSpace or DefaultExecutionSpace), which is exactly what create_executor
    // consumes. Crucially it threads the Kokkos execution-space stream, host executor and
    // memory-space-matched device allocator through to the Ginkgo executor, so Ginkgo runs on the
    // same stream Kokkos filled the matrix/RHS views on rather than on Ginkgo's default stream.
    return std::visit(
        [](auto concreteExec) -> std::shared_ptr<gko::Executor>
        {
            using ExecType = std::decay_t<decltype(concreteExec)>;
            return gko::ext::kokkos::create_executor(typename ExecType::exec {});
        },
        exec
    );
}

} // namespace

// Memoized: GinkgoSolver is reconstructed on every solve, and creating a fresh Ginkgo executor each
// time is wasteful -- e.g. a CUDA executor rebuilds its cuBLAS/cuSPARSE handles on construction.
// Return a stable executor per NeoN executor kind so every solve shares it. The cache is released
// in a Kokkos finalize hook so the executor is destroyed while the device is still alive.
//
// NOTE the cache is keyed by NeoN executor kind (variant index), not by device id, so a single
// process driving more than one device of the same kind would share one executor. That matches the
// usual one-device-per-rank HPC layout; revisit if intra-process multi-device support is added.
std::shared_ptr<gko::Executor> NeoN::la::ginkgo::getGkoExecutor(NeoN::Executor exec)
{
    // Register the finalize hook exactly once, on first use, so the cached executors are released
    // while the Kokkos device (and its CUDA context) is still alive.
    static std::once_flag hookFlag;
    std::call_once(
        hookFlag,
        []
        {
            Kokkos::push_finalize_hook(
                []
                {
                    for (auto& e : gkoExecutorCache())
                    {
                        e.reset();
                    }
                }
            );
        }
    );

    // Guard the lazy fill: magic-static initialisation protects the array's construction but not
    // these element writes, so a mutex is needed if executors are ever built from several threads.
    static std::mutex cacheMutex;
    const std::lock_guard<std::mutex> guard {cacheMutex};
    auto& cache = gkoExecutorCache();
    const std::size_t idx = exec.index();
    if (!cache[idx])
    {
        cache[idx] = createGkoExecutor(exec);
    }
    return cache[idx];
}

namespace
{

// Process-local cache of built solver factories, keyed by "<reuseKey>@<executorIndex>". A factory
// only encodes solver/preconditioner configuration (no matrix values), so reusing it across
// timesteps is numerically safe: generate() still rebuilds the preconditioner from the current
// matrix values on every solve. Released in a Kokkos finalize hook because each factory holds a
// reference to the (device) Ginkgo executor, which must be destroyed while the device is alive.
std::map<std::string, std::shared_ptr<const gko::LinOpFactory>>& gkoFactoryCache()
{
    static std::map<std::string, std::shared_ptr<const gko::LinOpFactory>> cache;
    return cache;
}

} // namespace

std::shared_ptr<const gko::LinOpFactory> NeoN::la::ginkgo::getOrBuildFactory(
    const Executor& exec,
    std::shared_ptr<const gko::Executor> gkoExec,
    const gko::config::pnode& config,
    const std::string& reuseKey
)
{
    auto build = [&]
    {
        return gko::config::parse(
                   config, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
        )
            .on(gkoExec);
    };

    // No stable key supplied (e.g. test code, or NeoFOAM did not thread a field name): preserve the
    // previous behaviour and build a fresh factory every time.
    if (reuseKey.empty())
    {
        return build();
    }

    static std::once_flag hookFlag;
    std::call_once(hookFlag, [] { Kokkos::push_finalize_hook([] { gkoFactoryCache().clear(); }); });

    static std::mutex cacheMutex;
    const std::lock_guard<std::mutex> guard {cacheMutex};
    auto& slot = gkoFactoryCache()[reuseKey + "@" + std::to_string(exec.index())];
    if (!slot)
    {
        slot = build();
    }
    return slot;
}


namespace NeoN::la::ginkgo
{

label computeNRows(const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys)
{
    return 3 * sys.rhs().size();
}

label computeNRows(const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys)
{
    return sys.rhs().size();
}

/*@brief create a new array by copying from view into ptr*/
template<typename T>
auto gkoCopyArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_const_array_view(exec, values.size(), values.data()).copy_to_array();
}

/* @brief create a ginkgo csr matrix by creating views into Csr<scalar> avoiding copies */
template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const CSRMatrix<scalar, IndexType>& mtx)
{
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), coeffsV.data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.colIdxs.size()), sparsityV.colIdxs.data()
    );
    auto row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.rowOffs.size()), sparsityV.rowOffs.data()
    );

    auto nrows = static_cast<gko::size_type>(mtx.nRows());
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
        exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const COOMatrix<scalar, IndexType>& mtx)
{
    const auto [coeffsV, sparsityV] = mtx.view();

    // NOTE we get a const view of the system but need a non const view to vals and indices
    auto vals = gko::array<scalar>::const_view(
        exec, static_cast<gko::size_type>(coeffsV.size()), coeffsV.data()
    );
    auto col = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(sparsityV.colIdxs.size()), sparsityV.colIdxs.data()
    );
    // sparsityV.rowOffs holds COO per-entry row indices; Ginkgo Csr::create_const needs
    // CSR row offsets (size nRows+1), which live in CooSparsityPattern::rowOffs_.
    const auto& csrRowOffs = mtx.rowOffs();
    auto row = gko::array<IndexType>::const_view(
        exec, static_cast<gko::size_type>(csrRowOffs.size()), csrRowOffs.view().data()
    );

    auto nrows = static_cast<gko::size_type>(csrRowOffs.size() - 1);
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create_const(
        exec, gko::dim<2> {nrows, nrows}, std::move(vals), std::move(col), std::move(row)
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor> exec, const CSRMatrix<Vec3, IndexType>& mtx)
{
    const auto rowsCopy = unpackRowOffs(mtx.rowOffs());
    const auto colsCopy = unpackColIdx(mtx.colIdxs(), rowsCopy, mtx.rowOffs());
    const auto valuesCopy = unpackMtxValues(mtx.values(), mtx.rowOffs(), rowsCopy);

    auto nrows = static_cast<gko::size_type>(3 * mtx.nRows());
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, nrows},
        gkoCopyArray(exec, valuesCopy.view()),
        gkoCopyArray(exec, colsCopy.view()),
        gkoCopyArray(exec, rowsCopy.view())
    ));
}

template<typename IndexType>
std::shared_ptr<const gko::LinOp>
createGkoMtxImpl(std::shared_ptr<const gko::Executor>, const COOMatrix<Vec3, IndexType>&)
{
    NF_THROW("createGkoMtxImpl: COOMatrix<Vec3> is not supported");
}

template<typename NeoNMatrixType>
std::shared_ptr<const gko::LinOp> createGkoMtx(const NeoNMatrixType& mtx)
{
    auto exec = getGkoExecutor(mtx.exec());
    return createGkoMtxImpl(exec, mtx);
}


template<typename VectorType>
SolverStatsEntry solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const VectorType& rhs,
    VectorType& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    // Non-owning: the caller keeps the solver alive so it can outlive apply() and have its
    // scratch Workspace extracted afterwards (Ginkgo PR #2000 reuse path). A prvalue/xvalue
    // bound here still lives for the duration of the call, so the existing call sites that pass
    // a temporary generate() result are unaffected.
    const std::unique_ptr<gko::LinOp>& solver,
    const L1ResidualControl* l1Control = nullptr
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, rhs.data(), nrows);
    auto x = gkoVecView(exec, xIn.data(), nrows);

    // L1-scaled residual path: stop and report on the scaled residual.
    if (l1Control != nullptr)
    {
        auto l1Res = solveWithL1Stop(exec, mtx, b, x, solver.get(), *l1Control);
        exec->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;
        return {static_cast<label>(l1Res.numIter), l1Res.initResNorm, l1Res.finalResNorm, duration};
    }

    // copy of rhs to compute the initial residual inline
    auto rhsCopy = VectorType(rhs);
    auto res = gkoVecView(exec, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    auto init = gko::initialize<vec>({0.0}, exec);
    res->compute_norm2(init);
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

SolverStats solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const Vector<Vec3>& rhs,
    Vector<Vec3>& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver
)
{
    exec->synchronize();
    auto startEval = std::chrono::steady_clock::now();

    using vec = gko::matrix::Dense<scalar>;
    label nrows = rhs.size();
    const auto b = gkoVecView(exec, rhs.data(), nrows); // [nrows x 3]
    auto x = gkoVecView(exec, xIn.data(), nrows);       // [nrows x 3]

    auto rhsCopy = Vector<Vec3>(rhs);
    auto res = gkoVecView(exec, rhsCopy.data(), nrows);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    mtx->apply(one, x, neg_one, res);

    // compute_norm2 on [nrows x 3] writes a [1 x 3] result — one L2 norm per column.
    auto colNorms = [&](std::shared_ptr<gko::matrix::Dense<scalar>> v) -> std::array<scalar, 3>
    {
        auto nv = vec::create(exec, gko::dim<2> {1, 3});
        v->compute_norm2(nv);
        auto nh = vec::create(exec->get_master(), gko::dim<2> {1, 3});
        nh->copy_from(nv);
        return {nh->at(0, 0), nh->at(0, 1), nh->at(0, 2)};
    };
    auto initNorms = colNorms(res);

    std::shared_ptr<const gko::log::Convergence<scalar>> logger =
        gko::log::Convergence<scalar>::create();
    solver->add_logger(logger);
    solver->apply(b, x);

    auto rhsCopyFinal = Vector<Vec3>(rhs);
    auto resFinal = gkoVecView(exec, rhsCopyFinal.data(), nrows);
    mtx->apply(one, x, neg_one, resFinal);
    auto finalNorms = colNorms(resFinal);

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


#ifdef NEON_GINKGO_PUBLIC_WORKSPACE
namespace
{

// Process-local cache of reusable solver scratch Workspaces (Ginkgo PR #2000), keyed like the
// factory cache. A Workspace is value-independent (sized by matrix dimension), so reusing it across
// solves of the same field is safe; cleared in a Kokkos finalize hook because it holds an executor
// reference that must die while the device is alive.
std::map<std::string, std::unique_ptr<gko::solver::Workspace>>& gkoWorkspaceCache()
{
    static std::map<std::string, std::unique_ptr<gko::solver::Workspace>> cache;
    return cache;
}

std::mutex& gkoWorkspaceCacheMutex()
{
    static std::mutex m;
    return m;
}

// Generate a solver, reusing the cached scratch Workspace for `wsKey` when non-empty. The
// preconditioner is still rebuilt from the current matrix values inside generate() — only the
// scratch memory is reused, so this does not introduce the stale-preconditioner problem.
std::unique_ptr<gko::LinOp> generateReusingWorkspace(
    const std::shared_ptr<const gko::LinOpFactory>& factory,
    std::shared_ptr<const gko::LinOp> mtx,
    const std::shared_ptr<const gko::Executor>& gkoExec,
    const std::string& wsKey
)
{
    if (wsKey.empty())
    {
        return factory->generate(std::move(mtx));
    }

    static std::once_flag hookFlag;
    std::call_once(
        hookFlag, [] { Kokkos::push_finalize_hook([] { gkoWorkspaceCache().clear(); }); }
    );

    std::unique_ptr<gko::solver::Workspace> ws;
    {
        const std::lock_guard<std::mutex> guard {gkoWorkspaceCacheMutex()};
        auto it = gkoWorkspaceCache().find(wsKey);
        if (it != gkoWorkspaceCache().end())
        {
            ws = std::move(it->second);
            gkoWorkspaceCache().erase(it);
        }
    }
    if (!ws)
    {
        ws = gko::solver::Workspace::create(gkoExec);
    }
    return factory->generate(std::move(mtx), std::move(ws));
}

// Extract the scratch Workspace back out of `solver` (consuming it) and return it to the cache so
// the next solve of the same field reuses the allocations. No-op (solver destroyed normally) when
// `wsKey` is empty.
void stashWorkspace(const std::string& wsKey, std::unique_ptr<gko::LinOp>&& solver)
{
    if (wsKey.empty())
    {
        return;
    }
    auto ws = gko::solver::invalidate_and_extract_workspace(std::move(solver));
    const std::lock_guard<std::mutex> guard {gkoWorkspaceCacheMutex()};
    gkoWorkspaceCache()[wsKey] = std::move(ws);
}

} // namespace
#endif // NEON_GINKGO_PUBLIC_WORKSPACE

SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    Kokkos::Profiling::pushRegion("ginkgo::mtxConvert");
    auto gkoMtx = createGkoMtx(sys.matrix());
    Kokkos::Profiling::popRegion();

    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;

    // "ginkgo::generate" isolates the solver+preconditioner build (the per-solve setup cost) from
    // the apply, which solve_impl times separately and reports as "Solve time". Together with the
    // factory reuse above, this is what lets you see whether the pressure phase is setup- or
    // apply-bound in the kp_space_time_stack Regions view.
    Kokkos::Profiling::pushRegion("ginkgo::generate");
#ifdef NEON_GINKGO_PUBLIC_WORKSPACE
    // Ginkgo PR #2000 (public workspace): reuse a per-(field,executor) scratch Workspace across
    // generate() calls so the value-independent Krylov/preconditioner scratch is allocated once
    // rather than every solve. Scoped to this scalar rank-local path with a non-empty reuseKey;
    // an empty key falls back to a plain generate. The solver is kept alive past solve_impl so the
    // workspace can be extracted back into the cache below.
    const std::string wsKey =
        reuseKey_.empty() ? std::string {} : reuseKey_ + "@" + std::to_string(exec_.index());
    auto solver = generateReusingWorkspace(factory_, gkoMtx, gkoExec_, wsKey);
#else
    auto solver = factory_->generate(gkoMtx);
#endif
    Kokkos::Profiling::popRegion();

    auto stats = solve_impl(gkoExec_, sys.rhs(), x, gkoMtx, solver, l1Control);
#ifdef NEON_GINKGO_PUBLIC_WORKSPACE
    stashWorkspace(wsKey, std::move(solver));
#endif
    return {stats};
}

/* @brief create a ginkgo csr matrix by unpacking and copying the Csr<Vec3> input */
template<typename IndexType>
std::shared_ptr<const gko::matrix::Csr<scalar, IndexType>> createGkoMtxImpl(
    std::shared_ptr<const gko::Executor> exec,
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, IndexType>>& sys
)
{
    // NOTE we get a const view of the system but need a non const view to vals and indices
    const auto mtx = sys.matrix();
    const auto rowsCopy = unpackRowOffs(mtx.rowOffs());
    const auto colsCopy = unpackColIdx(mtx.colIdxs(), rowsCopy, mtx.rowOffs());
    const auto valuesCopy = unpackMtxValues(mtx.values(), mtx.rowOffs(), rowsCopy);

    auto nrows = static_cast<gko::size_type>(computeNRows(sys));
    return gko::share(gko::matrix::Csr<scalar, IndexType>::create(
        exec,
        gko::dim<2> {nrows, nrows},
        gkoCopyArray(exec, valuesCopy.view()),
        gkoCopyArray(exec, colsCopy.view()),
        gkoCopyArray(exec, rowsCopy.view())
    ));
}

// wrapper to solve a single component of a <vec3> equation
template<unsigned int I>
void solveComponent(
    auto& sys, auto& x, auto& exec, auto& factory, auto& stats, const L1ResidualControl* l1Control
)
{
    auto rhs = getComponent<I>(sys.rhs());
    auto xcopy = getComponent<I>(x);
    auto values = getComponent<I>(sys.matrix().values());
    auto sparsity = sys.matrix().sparsity();
    auto mtx = CSRMatrix<scalar, localIdx> {values, sparsity};
    Kokkos::Profiling::pushRegion("ginkgo::mtxConvert");
    auto gkoMtx = createGkoMtx(mtx);
    Kokkos::Profiling::popRegion();
    // One generate() per velocity component: this region accumulates all three per momentum solve,
    // exposing the segregated-solve setup cost called out in the profiling notes.
    Kokkos::Profiling::pushRegion("ginkgo::generate");
    auto solver = factory->generate(gkoMtx);
    Kokkos::Profiling::popRegion();
    stats.entries.push_back(solve_impl(exec, rhs, xcopy, gkoMtx, solver, l1Control));
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solve(
    const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
) const
{
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    if (coupled_)
    {
        const auto gkoMtx = createGkoMtx(sys.matrix());
        auto rhsCopy = unpackVecValues(sys.rhs());
        auto xCopy = unpackVecValues(x);

        auto stats =
            solve_impl(gkoExec_, rhsCopy, xCopy, gkoMtx, factory_->generate(gkoMtx), l1Control);

        packVecValues(xCopy, x);
        return {stats};
    }
    else
    {
        auto stats = SolverStats {};
        solveComponent<0>(sys, x, gkoExec_, factory_, stats, l1Control);
        solveComponent<1>(sys, x, gkoExec_, factory_, stats, l1Control);
        solveComponent<2>(sys, x, gkoExec_, factory_, stats, l1Control);
        return stats;
    }
}


SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>& sys,
    Vector<Vec3>& x
) const
{
    auto gkoMtx = createGkoMtx(sys.matrix());
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    if (l1Control)
    {
        gkoExec_->synchronize();
        auto startEval = std::chrono::steady_clock::now();

        label nrows = sys.rhs().size();
        const auto b = gkoVecView(gkoExec_, sys.rhs().data(), nrows); // [nrows x 3]
        auto xView = gkoVecView(gkoExec_, x.data(), nrows);           // [nrows x 3]
        auto solver = factory_->generate(gkoMtx);
        auto l1Res = solveWithL1Stop(gkoExec_, gkoMtx, b, xView, solver.get(), *l1Control);

        gkoExec_->synchronize();
        auto endEval = std::chrono::steady_clock::now();
        auto duration =
            static_cast<scalar>(
                std::chrono::duration_cast<std::chrono::microseconds>(endEval - startEval).count()
            )
            / 1000.0;

        SolverStats stats;
        for (int i = 0; i < 3; ++i)
            stats.entries.push_back(
                {l1Res.numIter, l1Res.perColInitNorms[i], l1Res.perColFinalNorms[i], duration}
            );
        return stats;
    }
    using vec = gko::matrix::Dense<scalar>;
    label nrows = sys.rhs().size();
    // Mutable [nrows x 3] Dense view of x — write-through to x.data()
    auto xDense = gkoVecView(gkoExec_, x.data(), nrows);
    const scalar* rhsScalar = reinterpret_cast<const scalar*>(sys.rhs().data());

    gkoExec_->synchronize();
    SolverStats stats;
    for (gko::size_type col = 0; col < 3; ++col)
    {
        auto t0 = std::chrono::steady_clock::now();
        gko::span spanAll {gko::size_type {0}, static_cast<gko::size_type>(nrows)};
        gko::span spanC {col, col + 1};

        // [nrows x 1] const strided Dense view of rhs column col (no copy)
        // Data at rhsScalar+col, stride 3; array size = 3*nrows-col covers all nrows elements
        auto b_col = gko::share(vec::create_const(
            gkoExec_,
            gko::dim<2> {static_cast<gko::size_type>(nrows), 1},
            gko::array<scalar>::const_view(
                gkoExec_, static_cast<gko::size_type>(3 * nrows - col), rhsScalar + col
            ),
            3
        ));

        // [nrows x 1] mutable strided Dense view of x column col — write-through to x.data()
        auto x_col = xDense->create_submatrix(spanAll, spanC);

        auto initNorm_v = vec::create(gkoExec_, gko::dim<2> {1, 1});
        b_col->compute_norm2(initNorm_v); // initResNorm = ||b_col||₂  (x starts at 0)
        scalar initResNorm = retrieve(initNorm_v);

        auto solver = factory_->generate(gkoMtx);
        std::shared_ptr<const gko::log::Convergence<scalar>> logger =
            gko::log::Convergence<scalar>::create();
        solver->add_logger(logger);
        solver->apply(b_col, x_col);

        scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));
        auto numIter = label(logger->get_num_iterations());
        gkoExec_->synchronize();
        auto duration = static_cast<scalar>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - t0
                        )
                                                .count())
                      / 1000.0;
        stats.entries.push_back({numIter, initResNorm, finalResNorm, duration});
    }
    return stats;
}

template std::shared_ptr<const gko::LinOp>
createGkoMtx<CSRMatrix<scalar, localIdx>>(const CSRMatrix<scalar, localIdx>&);

template std::shared_ptr<const gko::LinOp>
createGkoMtx<COOMatrix<scalar, localIdx>>(const COOMatrix<scalar, localIdx>&);

}

#endif
