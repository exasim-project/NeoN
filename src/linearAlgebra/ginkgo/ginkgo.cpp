// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <array>
#include <sstream>
#include <unordered_map>

#include "NeoN/linearAlgebra/ginkgo.hpp"
#include "NeoN/linearAlgebra/preconditioner/aDIC.hpp"
#include "NeoN/linearAlgebra/preconditioner/adicGinkgo.hpp"
#include "NeoN/core/vector/vectorFreeFunctions.hpp"

gko::config::pnode NeoN::la::ginkgo::parse(const Dictionary& dictIn)
{
    Dictionary dict = dictIn;
    // Remove the top-level 'solver Ginkgo;' entry. Guard the cast on the value type:
    // parse() recurses into nested sub-dictionaries (below), and a nested Ginkgo config
    // can legitimately carry a 'solver' key whose value is itself a factory sub-dictionary
    // rather than the "Ginkgo" string -- e.g. the inner relaxation solver of an Ir apply
    // solver used as an Ic/Ilu l_solver. An unguarded any_cast<std::string> on such a node
    // throws bad_any_cast. Only the string-valued "Ginkgo" entry should be stripped here;
    // a dictionary-valued 'solver' is passed through to the recursive parse at the bottom.
    if (dict.contains("solver") && dict.isType<std::string>("solver")
        && dict.get<std::string>("solver") == "Ginkgo")
    {
        dict.remove("solver");
    }

    if (dict.contains("coupled"))
    {
        dict.remove("coupled");
    }

    // 'negateSystem' steers the solver to solve (-A) x = (-b) (handled in GinkgoSolver::solve);
    // it is not a Ginkgo config key.
    if (dict.contains("negateSystem"))
    {
        dict.remove("negateSystem");
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

    // 'checkFrequency' controls how often the L1 criterion evaluates the true residual
    // (handled via readL1ResidualControl); it is not a Ginkgo config key.
    if (dict.contains("checkFrequency"))
    {
        dict.remove("checkFrequency");
    }

    // 'preconReuse' (regeneration interval) and 'reuseKey' (cache key) steer preconditioner reuse
    // in GinkgoSolver; they are not Ginkgo config keys.
    for (const auto& key : {std::string("preconReuse"), std::string("reuseKey")})
    {
        if (dict.contains(key))
        {
            dict.remove(key);
        }
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
    return std::visit(
        [](auto concreteExec) -> std::shared_ptr<gko::Executor>
        {
            using ExecType = std::decay_t<decltype(concreteExec)>;
            if constexpr (std::is_same_v<ExecType, NeoN::SerialExecutor>)
            {
                return gko::ReferenceExecutor::create();
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::CPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_OMP)
                return gko::OmpExecutor::create();
#elif defined(KOKKOS_ENABLE_THREADS)
                return gko::ReferenceExecutor::create();
#endif
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::GPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_CUDA)
                return gko::CudaExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#elif defined(KOKKOS_ENABLE_HIP)
                return gko::HipExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#elif defined(KOKKOS_ENABLE_SYCL)
                return gko::DpcppExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#endif
                throw std::runtime_error {"No valid GPU executor mapping available"};
            }
            else
            {
                throw std::runtime_error {"Unsupported executor type"};
            }
            return gko::ReferenceExecutor::create();
        },
        exec
    );
}

} // namespace

// TODO: check if this can be replaced by Ginkgos executor mapping
//
// Memoized: GinkgoSolver is reconstructed on every solve, and creating a fresh Ginkgo (e.g. CUDA)
// executor each time is both wasteful and breaks preconditioner reuse -- a cached preconditioner is
// bound to the executor it was built on, so a per-solve executor makes Ginkgo clone (and on aDIC,
// crash) it onto the new executor every reuse. Returning a stable executor per NeoN executor kind
// keeps cached preconditioners valid. The cache is released in a Kokkos finalize hook so the
// executor is destroyed while the device is still alive.
std::shared_ptr<gko::Executor> NeoN::la::ginkgo::getGkoExecutor(NeoN::Executor exec)
{
    static bool hookRegistered = false;
    if (!hookRegistered)
    {
        Kokkos::push_finalize_hook(
            []()
            {
                for (auto& e : gkoExecutorCache())
                {
                    e.reset();
                }
            }
        );
        hookRegistered = true;
    }

    auto& cache = gkoExecutorCache();
    const std::size_t idx = exec.index();
    if (!cache[idx])
    {
        cache[idx] = createGkoExecutor(exec);
    }
    return cache[idx];
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


namespace
{

// One reused preconditioner per cache key (field name). The preconditioner is regenerated only
// every `preconReuse` solves; in between it is reused as a "frozen" (approximate) preconditioner.
struct PrecondCacheEntry
{
    std::shared_ptr<gko::LinOp> precond;
    std::uintptr_t sparsitySig; //!< rowOffs buffer address: changes when the mesh/sparsity changes
    int counter;                //!< solves seen for this key (drives the regeneration interval)
};

std::unordered_map<std::string, PrecondCacheEntry>& precondCache()
{
    static std::unordered_map<std::string, PrecondCacheEntry> cache;
    static bool hookRegistered = false;
    if (!hookRegistered)
    {
        // Release cached Ginkgo LinOps before Kokkos (and the device) is torn down. A Kokkos
        // finalize hook runs at the start of Kokkos::finalize(), avoiding the static-destruction
        // ordering crash that a plain static map of device-backed objects would risk.
        Kokkos::push_finalize_hook([]() { precondCache().clear(); });
        hookRegistered = true;
    }
    return cache;
}

} // namespace


GinkgoSolver::GinkgoSolver(Executor exec, const Dictionary& solverConfig) : Base(exec)
{
    gkoExec_ = getGkoExecutor(exec);
    coupled_ = solverConfig.get("coupled", false);
    negateSystem_ = solverConfig.get("negateSystem", false);
    l1Control_ = readL1ResidualControl(solverConfig);
    preconReuse_ = solverConfig.get("preconReuse", 1);
    reuseKey_ = solverConfig.get("reuseKey", std::string("default"));

    const std::string precondType =
        solverConfig.isDict("preconditioner")
                && solverConfig.subDict("preconditioner").contains("type")
            ? solverConfig.subDict("preconditioner").get<std::string>("type")
            : std::string {};
    useADIC_ = precondType == "aDIC";
    useADICGinkgo_ = precondType == "aDICGinkgo";
    injectPrecond_ = useADIC_ || useADICGinkgo_ || preconReuse_ > 1;

    // Default factory for the Vec3 / distributed / non-injected paths. The aDIC marker is stripped
    // first because Ginkgo's config parser cannot instantiate it (it is injected per solve); the
    // resulting factory is then unpreconditioned, which only ever serves Vec3 paths (aDIC is mapped
    // for the scalar pressure solve only).
    Dictionary cfgForFactory = solverConfig;
    if ((useADIC_ || useADICGinkgo_) && cfgForFactory.contains("preconditioner"))
    {
        cfgForFactory.remove("preconditioner");
    }
    config_ = parse(cfgForFactory);
    factory_ = gko::config::parse(
                   config_, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
    )
                   .on(gkoExec_);

    if (injectPrecond_)
    {
        // Config-built preconditioner reuse: keep the preconditioner sub-config so it can be
        // (re)generated from the current matrix and cached. The aDIC markers carry no Ginkgo
        // config.
        if (!useADIC_ && !useADICGinkgo_ && solverConfig.isDict("preconditioner"))
        {
            precondConfig_ = parse(solverConfig.subDict("preconditioner"));
        }
        // Solver config that references the injected preconditioner by registry key.
        Dictionary cfgInjected = solverConfig;
        if (cfgInjected.contains("preconditioner"))
        {
            cfgInjected.remove("preconditioner");
        }
        cfgInjected.insert("generated_preconditioner", std::string("neonGenPrecond"));
        injectedSolverConfig_ = parse(cfgInjected);
    }
}


std::unique_ptr<gko::LinOp> GinkgoSolver::generateInjectedSolver(
    std::shared_ptr<const gko::LinOp> gkoMtx, const CSRMatrix<scalar, localIdx>& neonMtx
) const
{
    auto& cache = precondCache();
    const auto sig = reinterpret_cast<std::uintptr_t>(neonMtx.rowOffs().data());

    auto it = cache.find(reuseKey_);
    const bool present = (it != cache.end());
    const int counter = present ? it->second.counter : 0;
    const bool regenerate =
        !present || (it->second.sparsitySig != sig) || (counter % std::max(preconReuse_, 1) == 0);

    std::shared_ptr<gko::LinOp> precond;
    if (regenerate)
    {
        if (useADIC_)
        {
            precond = std::make_shared<ADICPreconditioner>(gkoExec_, exec_, neonMtx);
        }
        else if (useADICGinkgo_)
        {
            // Ginkgo-native aDIC: build from the Ginkgo CSR (gkoMtx is that matrix). The
            // preconditioner deep-copies it internally, so the cached instance stays frozen.
            auto csr = gko::as<const gko::matrix::Csr<scalar, localIdx>>(gkoMtx);
            precond = gko::share(ADICGinkgoPreconditioner::create(gkoExec_, csr));
        }
        else
        {
            auto precondFactory = gko::config::parse(
                                      precondConfig_,
                                      gko::config::registry(),
                                      gko::config::make_type_descriptor<scalar>()
            )
                                      .on(gkoExec_);
            precond = std::shared_ptr<gko::LinOp>(precondFactory->generate(gkoMtx));
        }
        cache[reuseKey_] = {precond, sig, counter + 1};
    }
    else
    {
        precond = it->second.precond;
        it->second.counter = counter + 1;
    }

    // Build the Krylov solver referencing the (cached) preconditioner via the registry, reusing the
    // full solver configuration (type, criteria, ...) parsed once in the constructor.
    gko::config::registry reg;
    reg.emplace("neonGenPrecond", precond);
    auto factory =
        gko::config::parse(injectedSolverConfig_, reg, gko::config::make_type_descriptor<scalar>())
            .on(gkoExec_);
    return factory->generate(gkoMtx);
}


template<typename VectorType>
SolverStatsEntry solve_impl(
    std::shared_ptr<const gko::Executor> exec,
    const VectorType& rhs,
    VectorType& xIn,
    std::shared_ptr<const gko::LinOp> mtx,
    std::unique_ptr<gko::LinOp> solver,
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

SolverStatsEntry solve_impl(
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
    const auto b = gkoVecView(exec, rhs.data(), nrows);
    auto x = gkoVecView(exec, xIn.data(), nrows);

    // create a copy of rhs so that we can inline compute
    // the residual
    auto rhsCopy = Vector<Vec3>(rhs);
    auto res = gkoVecView(exec, rhsCopy.data(), nrows);

    // compute Ax-b -> res
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


SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
) const
{
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;

    // Cholesky-type preconditioners (Ic/ParIc) require a positive-definite matrix, but the
    // OpenFOAM pressure Laplacian is assembled negative-(semi)definite. Present Ginkgo the
    // equivalent SPD system (-A) x = (-b): the solution x and the residual |b - A x| (hence the
    // L1-scaled residual) are unchanged. Negation is done on a deep copy so the caller's system
    // is untouched.
    // The native aDIC preconditioner and preconditioner reuse are routed through
    // generateInjectedSolver (the preconditioner is built/cached and injected as a generated
    // preconditioner); otherwise the per-solve factory_ path is unchanged.
    if (negateSystem_)
    {
        auto negSys = sys; // deep-copies values and rhs; shares the immutable sparsity pattern
        negSys.matrix().values() *= -1.0;
        negSys.rhs() *= -1.0;
        auto gkoMtx = createGkoMtx(negSys.matrix());
        auto solver = injectPrecond_ ? generateInjectedSolver(gkoMtx, negSys.matrix())
                                     : factory_->generate(gkoMtx);
        return {solve_impl(gkoExec_, negSys.rhs(), x, gkoMtx, std::move(solver), l1Control)};
    }

    auto gkoMtx = createGkoMtx(sys.matrix());
    auto solver =
        injectPrecond_ ? generateInjectedSolver(gkoMtx, sys.matrix()) : factory_->generate(gkoMtx);
    return {solve_impl(gkoExec_, sys.rhs(), x, gkoMtx, std::move(solver), l1Control)};
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
    auto gkoMtx = createGkoMtx(mtx);
    auto solver = factory->generate(gkoMtx);

    stats.entries.push_back(solve_impl(exec, rhs, xcopy, gkoMtx, std::move(solver), l1Control));
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
        auto solver = factory_->generate(gkoMtx);

        auto rhsCopy = unpackVecValues(sys.rhs());
        auto xCopy = unpackVecValues(x);

        auto stats = solve_impl(gkoExec_, rhsCopy, xCopy, gkoMtx, std::move(solver), l1Control);

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


template<unsigned int I>
void solveVec3RhsComponent(
    const Vector<Vec3>& rhs,
    Vector<Vec3>& x,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOpFactory> factory,
    std::shared_ptr<const gko::LinOp> gkoMtx,
    SolverStats& stats,
    const L1ResidualControl* l1Control
)
{
    auto rhsComp = getComponent<I>(rhs);
    auto xcopy = getComponent<I>(x);
    auto solver = factory->generate(gkoMtx);
    stats.entries.push_back(solve_impl(exec, rhsComp, xcopy, gkoMtx, std::move(solver), l1Control));
    setComponent<I>(xcopy, x);
}

SolverStats GinkgoSolver::solve(
    const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>& sys,
    Vector<Vec3>& x
) const
{
    auto stats = SolverStats {};
    auto gkoMtx = createGkoMtx(sys.matrix());
    const L1ResidualControl* l1Control = l1Control_ ? &l1Control_.value() : nullptr;
    solveVec3RhsComponent<0>(sys.rhs(), x, gkoExec_, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponent<1>(sys.rhs(), x, gkoExec_, factory_, gkoMtx, stats, l1Control);
    solveVec3RhsComponent<2>(sys.rhs(), x, gkoExec_, factory_, gkoMtx, stats, l1Control);
    return stats;
}

template std::shared_ptr<const gko::LinOp>
createGkoMtx<CSRMatrix<scalar, localIdx>>(const CSRMatrix<scalar, localIdx>&);

template std::shared_ptr<const gko::LinOp>
createGkoMtx<COOMatrix<scalar, localIdx>>(const COOMatrix<scalar, localIdx>&);

}

#endif
