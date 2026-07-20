// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NF_WITH_GINKGO

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
    std::shared_ptr<gko::matrix::Coo<scalar, IndexType>>& nonLocalMtxCache
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

/** @brief Result of a solve governed by the L1-scaled residual stopping criterion. */
struct L1ResidualResult
{
    gko::size_type numIter; //!< iterations performed
    scalar initResNorm;     //!< combined scaled L1 initial residual (sum of columns)
    scalar finalResNorm;    //!< combined scaled L1 final residual (sum of columns)
    // Per-column scaled residuals; populated only for multi-RHS (Vec3) solves (size == ncols).
    std::vector<scalar> perColInitNorms;
    std::vector<scalar> perColFinalNorms;
};

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

class GinkgoSolver : public SolverFactory::template Register<GinkgoSolver>
{

    using Base = SolverFactory::template Register<GinkgoSolver>;

public:

    GinkgoSolver(Executor exec, const Dictionary& solverConfig)
        : Base(exec), gkoExec_(getGkoExecutor(exec)), coupled_(solverConfig.get("coupled", false)),
          l1Control_(readL1ResidualControl(solverConfig)), config_(parse(solverConfig)),
          factory_(gko::config::parse(
                       config_, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
          )
                       .on(gkoExec_))
    {}

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
    bool coupled_; // whether to solve LinearSystem<Vec3> as one or three systems
    // L1-scaled residual stopping controls; set only when "l1ScaledResidual" is enabled
    std::optional<L1ResidualControl> l1Control_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
#ifdef NF_WITH_MPI_SUPPORT
    // Both caches are null until the first solve; after that topology is fixed.
    mutable std::shared_ptr<gko::experimental::distributed::index_map<label, gko::int64>>
        cachedImap_;
    mutable std::shared_ptr<gko::matrix::Coo<scalar, localIdx>> cachedNonLocalMtx_;
#endif
};


}

#endif
