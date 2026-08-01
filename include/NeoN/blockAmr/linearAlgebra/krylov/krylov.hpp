// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/stopNormInf.hpp"

namespace blockamr::la
{

/* @brief The stopping test one solve runs, as one argument: an iteration cap, a residual
 *        reduction by `reductionFactor` against `baseline` (rhs_norm for a relative rtol,
 *        absolute when the caller folded rtol in), an absolute `atol` when it is > 0, and the
 *        norm all of that is measured in -- "l2" (Ginkgo's) or "linf" (MLMG's, stopNormInf.hpp).
 *        `norm` carries its default here, so a caller that never names one stays a 4-field
 *        aggregate.
 */
struct StopSpec
{
    int maxIter;
    gko::stop::mode baseline;
    double reductionFactor;
    double atol;
    std::string norm = "l2";
};

// The (Iteration, ResidualNorm[, ResidualNorm]) chain every Krylov solve here uses.
inline std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
makeCriteria(std::shared_ptr<const gko::Executor> exec, const StopSpec& stop)
{
    const NormKind kind = parseNorm(stop.norm);
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
    criteria.push_back(gko::stop::Iteration::build()
                           .with_max_iters(static_cast<gko::size_type>(stop.maxIter))
                           .on(exec));
    auto residual = [&](gko::stop::mode base,
                        double factor) -> std::shared_ptr<const gko::stop::CriterionFactory>
    {
        if (kind == NormKind::linf)
        {
            return ResidualNormInf::build().with_baseline(base).with_reduction_factor(factor).on(
                exec
            );
        }
        return gko::stop::ResidualNorm<double>::build()
            .with_baseline(base)
            .with_reduction_factor(factor)
            .on(exec);
    };
    criteria.push_back(residual(stop.baseline, stop.reductionFactor));
    if (stop.atol > 0.0)
    {
        criteria.push_back(residual(gko::stop::mode::absolute, stop.atol));
    }
    return criteria;
}

// Build a Krylov solver over `op`, stopping on the `stop` chain above -- iteration count,
// ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() serves many right-hand sides)
// or, when atol > 0, ||r|| <= atol. A non-null `precond` becomes its generated preconditioner.
std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const StopSpec& stop,
    std::shared_ptr<const gko::LinOp> precond
);

// The cg/bicgstab/gmres subset used by the one-shot solves in oneshot.cpp. The criteria
// come from the caller, not makeCriteria: those stop against a baseline computed once from
// the ORIGINAL system. `what` names the caller in the "unknown solver" message.
std::shared_ptr<gko::LinOp> generateBasicSolver(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>& criteria,
    const char* what
);

// Fill the SolveResult every solve entry point returns. `diagnostic` stays empty here:
// its thresholds are calibrated for a V-cycle's roughly constant contraction.
inline SolveResult makeSolveResult(
    std::int64_t num_iters, double res_norm, bool converged, const std::vector<double>& res_history
)
{
    SolveResult r;
    r.num_iters = num_iters;
    r.res_norm = res_norm;
    r.converged = converged;
    r.res_history = res_history;
    // Geometric mean of the per-iteration reduction; the history holds size() - 1 of them.
    double contraction = 0.0;
    if (res_history.size() >= 2 && res_history.front() > 0.0)
    {
        contraction = std::pow(
            res_history.back() / res_history.front(),
            1.0 / static_cast<double>(res_history.size() - 1)
        );
    }
    r.contraction = contraction;
    r.diagnostic = "";
    return r;
}

// Counters from a gko::log::Convergence logger, history from a ResidualHistoryLogger.
inline SolveResult makeSolveResult(
    const gko::log::Convergence<double>& logger,
    const ResidualHistoryLogger& resLogger,
    double res_norm
)
{
    return makeSolveResult(
        static_cast<std::int64_t>(logger.get_num_iterations()),
        res_norm,
        logger.has_converged(),
        resLogger.history()
    );
}

} // namespace blockamr::la
