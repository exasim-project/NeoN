// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NF_WITH_GINKGO

#include <array>
#include <chrono>
#include <optional>
#include <string>

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>
#include <ginkgo/extensions/config/json_config.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"
#include "NeoN/linearAlgebra/ginkgo/mergedPgm.hpp" // MergedPgm mergeLevels-style coarsening


namespace NeoN::la::ginkgo
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec);

gko::config::pnode parse(const Dictionary& dict);

/** @brief create a ginkgo matrix by creating views */
template<typename NeoNMatrixType>
std::shared_ptr<const gko::LinOp> createGkoMtx(const NeoNMatrixType& mtx);

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/distributed/communicationPattern.hpp"

template<typename IndexType>
std::shared_ptr<const gko::LinOp> createGkoMtxDist(
    std::shared_ptr<const gko::Executor> exec,
    const gko::experimental::mpi::communicator& comm,
    const CSRMatrix<scalar, IndexType>& mtx,
    const COOMatrix<scalar, IndexType>& bmtx,
    const CommunicationPattern& commPattern,
    std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>& imapCache,
    std::shared_ptr<gko::matrix::Coo<scalar, IndexType>>& nonLocalMtxCache,
    const std::string& localMatrixFormat = "Csr"
);
#endif // NF_WITH_MPI_SUPPORT

template<typename T>
gko::array<T> gkoArrayView(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

/** @brief Non-owning mutable Dense view; 1 column for scalar*, 3 columns for Vec3* */
template<typename T>
std::shared_ptr<gko::matrix::Dense<scalar>>
gkoVecView(std::shared_ptr<const gko::Executor> exec, T* ptr, localIdx s)
{
    constexpr std::size_t cols = std::is_same_v<T, Vec3> ? 3 : 1;
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<scalar>::create(
        exec,
        gko::dim<2> {size, cols},
        gkoArrayView<scalar>(exec, std::span<scalar> {reinterpret_cast<scalar*>(ptr), cols * size}),
        cols
    ));
}

/** @brief Non-owning const Dense view; 1 column for scalar*, 3 columns for Vec3* */
template<typename T>
std::shared_ptr<const gko::matrix::Dense<scalar>>
gkoVecView(std::shared_ptr<const gko::Executor> exec, const T* ptr, localIdx s)
{
    constexpr std::size_t cols = std::is_same_v<T, Vec3> ? 3 : 1;
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<scalar>::create_const(
        exec,
        gko::dim<2> {size, cols},
        gko::array<scalar>::const_view(exec, cols * size, reinterpret_cast<const scalar*>(ptr)),
        cols
    ));
}

/** @brief retrieve a scalar value from a ginkgo dense matrix on any executor */
template<typename InType>
scalar retrieve(const InType& in)
{
    using vec = gko::matrix::Dense<scalar>;
    auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
    return host->copy_from(in)->at(0);
}

/** @brief Control parameters for the L1-scaled residual stopping criterion.
 *
 * An absolute tolerance and a relative tolerance (relative to the initial residual) on
 * the L1-scaled residual, plus iteration bounds.
 */
struct L1ResidualControl
{
    scalar tolerance;        //!< absolute tolerance on the scaled L1 residual
    scalar relTol;           //!< relative tolerance (vs. initial residual); 0 disables
    gko::size_type maxIter;  //!< maximum iteration count
    gko::size_type minIter;  //!< minimum iteration count before tolerances are tested
    localIdx checkFrequency; //!< evaluate the (expensive) true residual every N iterations; <=1 =
                             //!< every iteration
};

/** @brief Result of a solve governed by the L1-scaled residual stopping criterion.
 *
 * Doubles as the report sink for the config-registered criterion (see
 * makeL1CriterionFactory): @c fired tells the solve path whether the criterion actually ran
 * (and therefore whether initResNorm/finalResNorm/numIter hold scaled L1 values).
 */
struct L1ResidualResult
{
    gko::size_type numIter = 0; //!< iterations performed
    scalar initResNorm = 0.0;   //!< combined scaled L1 initial residual (sum of columns)
    scalar finalResNorm = 0.0;  //!< combined scaled L1 final residual (sum of columns)
    bool fired = false;         //!< set by the criterion on its first check (config path)
    // Per-column scaled residuals; populated only for multi-RHS (Vec3) solves (size == ncols).
    std::vector<scalar> perColInitNorms;
    std::vector<scalar> perColFinalNorms;
};

/** @brief Registry key under which the L1-scaled residual criterion is stored so a Ginkgo
 * configFile can reference it by name in its "criteria" array, e.g.
 *   "criteria": [ {"type": "Iteration", "max_iters": 150}, "neon::l1ScaledResidual" ]
 * The "neon::" prefix avoids clashing with Ginkgo's own config type names.
 */
inline constexpr const char* l1CriterionKey = "neon::l1ScaledResidual";

/** @brief Build the L1-scaled residual criterion as a Ginkgo CriterionFactory for storage in
 * a gko::config::registry, so it can be named from a configFile (@see l1CriterionKey).
 *
 * The matrix / RHS are taken from the CriterionArgs at solve time, so they need not be known
 * here. When @p report is non-null the criterion writes the scaled initial/final residual and
 * iteration count there. Defined in ginkgoL1Stop.cpp.
 */
std::shared_ptr<gko::stop::CriterionFactory> makeL1CriterionFactory(
    std::shared_ptr<const gko::Executor> exec,
    const L1ResidualControl& control,
    L1ResidualResult* report
);

/** @brief Solve @p solver with an L1-scaled residual stopping criterion attached.
 *
 * Attaches a criterion that stops on the L1-scaled residual sum|b - A x| / normFactor
 * and returns the scaled initial/final residual and iteration count. @p x is updated in
 * place. Defined in ginkgoL1StopSerial.cpp (serial / rank-local).
 */
L1ResidualResult solveWithL1Stop(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<const gko::matrix::Dense<scalar>> b,
    std::shared_ptr<gko::matrix::Dense<scalar>> x,
    gko::LinOp* solver,
    const L1ResidualControl& control
);

#ifdef NF_WITH_MPI_SUPPORT
/** @brief Distributed counterpart of solveWithL1Stop.
 *
 * Same L1-scaled residual stopping criterion, but over distributed vectors so the
 * norm factor and residual norms are reduced globally across ranks. Shares the
 * implementation with the serial overload (see ginkgoL1Stop.cpp).
 */
L1ResidualResult solveWithL1StopDist(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> mtx,
    std::shared_ptr<const gko::experimental::distributed::Vector<scalar>> b,
    std::shared_ptr<gko::experimental::distributed::Vector<scalar>> x,
    gko::LinOp* solver,
    const L1ResidualControl& control
);
#endif

/** @brief Read the L1-scaled residual stopping controls from a solver configuration.
 *
 * Returns std::nullopt unless the solver dictionary opts in via "l1ScaledResidual".
 * The flag is accepted as a bool, an int, or a truthy word/string ("true"/"yes"/"on"/"1")
 * so it works both programmatically and when read from a dictionary file (where a
 * boolean token arrives as a word). Tolerances and the iteration cap are read from the
 * "criteria" sub-dict (absolute_residual_norm / initial_residual_norm / iteration).
 */
// A boolean switch read from an OpenFOAM dictionary is tokenized as a word and stored as a
// std::string (or, for a numeric form, an int), not a bool -- reading it directly as bool throws
// bad_any_cast. Accept any of the representations the dict parser may produce.
inline bool readSwitch(const Dictionary& cfg, const std::string& key, bool defaultValue)
{
    if (!cfg.contains(key)) return defaultValue;
    if (cfg.isType<bool>(key)) return cfg.get<bool>(key);
    if (cfg.isType<int>(key)) return cfg.get<int>(key) != 0;
    if (cfg.isType<std::string>(key))
    {
        const std::string v = cfg.get<std::string>(key);
        return (v == "true" || v == "yes" || v == "on" || v == "1");
    }
    return defaultValue;
}

inline std::optional<L1ResidualControl> readL1ResidualControl(const Dictionary& cfg)
{
    const std::string flag = "l1ScaledResidual";
    if (!cfg.contains(flag))
    {
        return std::nullopt;
    }
    // A boolean read from a dictionary file is stored as a word/string, not a bool;
    // accept the common representations rather than assuming a single type.
    bool enabled = false;
    if (cfg.isType<bool>(flag))
    {
        enabled = cfg.get<bool>(flag);
    }
    else if (cfg.isType<int>(flag))
    {
        enabled = cfg.get<int>(flag) != 0;
    }
    else if (cfg.isType<std::string>(flag))
    {
        const std::string v = cfg.get<std::string>(flag);
        enabled = (v == "true" || v == "yes" || v == "on" || v == "1");
    }
    if (!enabled)
    {
        return std::nullopt;
    }

    L1ResidualControl control {0.0, 0.0, 1000, 0, 1};

    // criteria entries may be stored as int, label or scalar depending on source
    auto readScalar = [](const Dictionary& d, const std::string& key, scalar fallback)
    {
        if (!d.contains(key)) return fallback;
        if (d.isType<int>(key)) return scalar(d.get<int>(key));
        if (d.isType<label>(key)) return scalar(d.get<label>(key));
        return d.get<scalar>(key);
    };
    auto readInt = [](const Dictionary& d, const std::string& key, localIdx fallback)
    {
        if (!d.contains(key)) return fallback;
        if (d.isType<int>(key)) return localIdx(d.get<int>(key));
        if (d.isType<scalar>(key)) return localIdx(d.get<scalar>(key));
        return d.get<localIdx>(key);
    };

    // OpenFOAM-style top-level keys (the configFile solver path keeps them: there is no
    // mapped "criteria" subdict there). Read first so they seed the control...
    control.tolerance = readScalar(cfg, "tolerance", control.tolerance);
    control.relTol = readScalar(cfg, "relTol", control.relTol);
    control.maxIter = readInt(cfg, "maxIter", control.maxIter);

    // ...then let a mapped "criteria" subdict (Ginkgo-style keys) override when present,
    // preserving the existing behaviour for fvSolution-mapped solvers.
    if (cfg.contains("criteria"))
    {
        const Dictionary& criteria = cfg.subDict("criteria");
        control.tolerance = readScalar(criteria, "absolute_residual_norm", control.tolerance);
        control.relTol = readScalar(criteria, "initial_residual_norm", control.relTol);
        control.maxIter = readInt(criteria, "iteration", control.maxIter);
    }
    control.minIter = readInt(cfg, "minIter", control.minIter);
    control.checkFrequency = readInt(cfg, "checkFrequency", control.checkFrequency);

    return control;
}

/** @brief Recursively test whether a Ginkgo config tree references @p name as a (criterion)
 * string anywhere — used to detect a configFile that names the L1 criterion so the solver
 * can rely on the in-config criterion instead of attaching one post-hoc.
 */
inline bool pnodeReferencesString(const gko::config::pnode& node, const std::string& name)
{
    using tag = gko::config::pnode::tag_t;
    switch (node.get_tag())
    {
    case tag::string:
        return node.get_string() == name;
    case tag::array:
        for (const auto& e : node.get_array())
            if (pnodeReferencesString(e, name)) return true;
        return false;
    case tag::map:
        for (const auto& kv : node.get_map())
            if (pnodeReferencesString(kv.second, name)) return true;
        return false;
    default:
        return false;
    }
}

class GinkgoSolver : public SolverFactory::template Register<GinkgoSolver>
{

    using Base = SolverFactory::template Register<GinkgoSolver>;

public:

    GinkgoSolver(Executor exec, const Dictionary& solverConfig)
        : Base(exec), gkoExec_(getGkoExecutor(exec)),
          coupled_(readSwitch(solverConfig, "coupled", false)),
          l1Control_(readL1ResidualControl(solverConfig)), config_(parse(solverConfig)),
          localMatrixFormat_(solverConfig.get("localMatrixFormat", std::string("Csr"))),
          cacheSolver_(readSwitch(solverConfig, "cacheSolver", false)),
          preconditionerRebuildInterval_(
              static_cast<localIdx>(solverConfig.get("preconditionerRebuildInterval", 0))
          )
    {
        // Register NeoN's L1-scaled residual criterion in the Ginkgo config registry so a
        // configFile can name it (l1CriterionKey) in its "criteria" array. Only needed when
        // the L1 stop is requested (l1ScaledResidual); the factory carries the tolerances and
        // a report sink. If the parsed config actually references it, the criterion lives
        // INSIDE the built solver (l1InConfig_) and reports via l1Report_; otherwise the
        // existing post-hoc attach in solve() handles the flag-only case unchanged.
        gko::config::registry reg;
        if (l1Control_)
        {
            l1CritFactory_ = makeL1CriterionFactory(gkoExec_, *l1Control_, &l1Report_);
            reg.emplace(std::string(l1CriterionKey), l1CritFactory_);
            l1InConfig_ = pnodeReferencesString(config_, l1CriterionKey);
        }

        // Register NeoN's MergedPgm coarseners (mergeLevels-style: merge k Pgm steps into one level
        // via SpGEMM-composed prolongations). A configFile's `mg_level` can then name them, e.g.
        // "mg_level": ["neon::pgmMerge2"]. Named-registry pattern, same as the L1 criterion above.
        // pgmMerge1 = merge_levels 1 = plain Pgm (single coarsening step), but via the SAME
        // MergedPgm code path as 2/3 -- so the cache (update_matrix_value) and distributed branches
        // behave identically across the merge-levels sweep, unlike swapping in native gko
        // multigrid::Pgm.
        reg.emplace("neon::pgmMerge1", makeMergedPgmFactory<scalar>(gkoExec_, 1));
        reg.emplace("neon::pgmMerge2", makeMergedPgmFactory<scalar>(gkoExec_, 2));
        reg.emplace("neon::pgmMerge3", makeMergedPgmFactory<scalar>(gkoExec_, 3));
        reg.emplace("neon::pgmMerge4", makeMergedPgmFactory<scalar>(gkoExec_, 4));

        // Mixed-precision: parse the user's solver config in lower precision and wrap it in an
        // outer fp64 Ir (Iterative Refinement) solver. The outer IR maintains the solution and
        // residual in fp64; the inner correction solve runs in the reduced precision, which is
        // faster on GPU. irIterations controls how many outer refinement steps are taken (1 is
        // sufficient for most CFD use cases where per-step accuracy requirements are modest).
        // Incompatible with l1ScaledResidual-in-config because l1 criterion is fp64-typed and
        // cannot be embedded in the lower-precision inner factory.
        const auto innerPrecision = solverConfig.get("innerPrecision", std::string(""));
        if (!innerPrecision.empty())
        {
            if (l1InConfig_)
                NF_ERROR_EXIT("innerPrecision cannot be combined with l1ScaledResidual in config");
            const auto irIterations =
                static_cast<gko::size_type>(solverConfig.get("irIterations", 1));
            auto buildIR = [&](gko::config::type_descriptor innerTypeDesc)
            {
                auto innerFactory = gko::config::parse(config_, reg, innerTypeDesc).on(gkoExec_);
                factory_ =
                    gko::solver::Ir<scalar>::build()
                        .with_solver(std::move(innerFactory))
                        .with_criteria(gko::stop::Iteration::build().with_max_iters(irIterations))
                        .on(gkoExec_);
            };
            if (innerPrecision == "float") buildIR(gko::config::make_type_descriptor<float>());
#if GINKGO_ENABLE_BFLOAT16
            else if (innerPrecision == "bfloat16")
                buildIR(gko::config::make_type_descriptor<gko::bfloat16>());
#endif
            else
            {
                const char* validOptions = "\"float\""
#if GINKGO_ENABLE_BFLOAT16
                                           " or \"bfloat16\""
#endif
                    ;
                NF_ERROR_EXIT(
                    "Unknown innerPrecision '" << innerPrecision << "': use " << validOptions
                );
            }
        }
        else
        {
            factory_ = gko::config::parse(config_, reg, gko::config::make_type_descriptor<scalar>())
                           .on(gkoExec_);
        }
    }

    static std::string name() { return "Ginkgo"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual SolverStats solve(
        const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
    ) const final;

    virtual SolverStats solve(
        const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
    ) const final;

    virtual SolverStats solve(
        const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>&
            sys,
        Vector<Vec3>& x
    ) const final;

#ifdef NF_WITH_MPI_SUPPORT
    virtual SolverStats solveDist(
        const LinearSystem<scalar, scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
    ) const final;

    virtual SolverStats solveDist(
        const LinearSystem<Vec3, Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
    ) const final;

    virtual SolverStats solveDist(
        const LinearSystem<scalar, Vec3, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>>&
            sys,
        Vector<Vec3>& x
    ) const final;
#endif

    // TODO why use a smart pointer here?
    virtual std::unique_ptr<SolverFactory> clone() const final
    {
        NF_ERROR_EXIT("Not implemented");
        return {};
    }

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    bool coupled_;
    std::optional<L1ResidualControl> l1Control_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
    std::string localMatrixFormat_;
    // L1-scaled residual criterion registered into the config registry (l1Control_ set).
    std::shared_ptr<gko::stop::CriterionFactory> l1CritFactory_;
    // True when config_ names the L1 criterion, so it is built into factory_'s solver and
    // reports through l1Report_ (the in-config path); false keeps the post-hoc attach.
    bool l1InConfig_ = false;
    // Report sink the in-config criterion writes its scaled L1 residual / iters into.
    mutable L1ResidualResult l1Report_;
    // Solver reuse across timesteps (Strategy 1b, see
    // docs/plans/ginkgo-solver-reuse-and-shared-allocator.md in NeoFOAM). When enabled the
    // generated solver is cached and, on later solves with unchanged matrix structure, refreshed in
    // place via gko::UpdateMatrixValue (the multigrid Pgm aggregation + smoother setup are reused)
    // instead of regenerating the whole hierarchy. Indexed by cache slot: [0] for scalar /
    // coupled-Vec3-rhs systems. Structure key = {nRows, localNnz, nonLocalNnz}; a mismatch drops
    // the cache and regenerates. Wired for the scalar distributed (pressure) solveDist only; the
    // segregated / implicit-transform Vec3 paths regenerate (cudafe++ gko::LinOp signature
    // limitation). mutable because solve() is const.
    //
    // cacheSolver_: opt-in via the "cacheSolver" dict entry (default off -> regenerate every solve,
    // i.e. the original behaviour). preconditionerRebuildInterval_: "preconditionerRebuildInterval"
    // dict entry; when > 0 the cached solver is fully regenerated (preconditioner rebuilt from
    // scratch) every Nth solve instead of updated in place, so Pgm aggregation drift is bounded;
    // 0 updates in place indefinitely. cachedSolveCount_: solves served by the current cached
    // solver per slot, reset on each (re)generate.
    bool cacheSolver_;
    localIdx preconditionerRebuildInterval_;
    mutable std::array<std::shared_ptr<gko::LinOp>, 3> cachedSolver_;
    mutable std::array<std::array<gko::size_type, 3>, 3> cachedSolverStructure_ {};
    mutable std::array<localIdx, 3> cachedSolveCount_ {};
#ifdef NF_WITH_MPI_SUPPORT
    // Both caches are null until the first solve; after that topology is fixed.
    mutable std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>
        cachedImap_;
    mutable std::shared_ptr<gko::matrix::Coo<scalar, localIdx>> cachedNonLocalMtx_;
#endif
};


}

#endif
