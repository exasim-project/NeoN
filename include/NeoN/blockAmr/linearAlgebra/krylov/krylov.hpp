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

// The (Iteration, ResidualNorm[, ResidualNorm]) stopping-criteria chain shared by
// every Krylov solve here: stop on `max_iter` iterations, or on a ResidualNorm with
// the given `baseline` (gko::stop::mode::rhs_norm for a relative rtol against
// ||rhs||; ::absolute when the caller already folded rtol into an absolute
// `reduction_factor`), plus — when atol > 0 — a second criterion whose baseline is
// always absolute, for the plain ||r|| <= atol test every call site wants. atol = 0.0
// reproduces the call sites that never had an atol branch.
//
// `norm` selects which norm the residual criteria measure: "l2" (Ginkgo's
// gko::stop::ResidualNorm, the default) or "linf" (ResidualNormInf, MLMG's norm — see
// stopNormInf.hpp). The iteration criterion is norm-independent.
inline std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> makeCriteria(
    std::shared_ptr<const gko::Executor> exec,
    int max_iter,
    gko::stop::mode baseline,
    double reduction_factor,
    double atol,
    const std::string& norm = "l2"
)
{
    const NormKind kind = parseNorm(norm);
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
    criteria.push_back(
        gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(max_iter)).on(exec)
    );
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
    criteria.push_back(residual(baseline, reduction_factor));
    if (atol > 0.0)
    {
        criteria.push_back(residual(gko::stop::mode::absolute, atol));
    }
    return criteria;
}

// Build a Krylov solver over `op`, stopping on iteration count, on the relative
// residual ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() is reused
// across right-hand sides), or — when atol > 0 — on the absolute ||r|| <= atol. A
// non-null `precond` (an already-generated LinOp, e.g. MlmgPrecond) becomes the
// solver's generated preconditioner. `norm` picks the norm both residual tests
// measure in ("l2" | "linf"); neither it nor `precond` has a default because
// KrylovSolver::build, their one caller, always passes both.
std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol,
    double atol,
    std::shared_ptr<const gko::LinOp> precond,
    const std::string& norm
);

// The cg/bicgstab/gmres subset used by the one-shot (non-persistent) solves in
// oneshot.cpp. The criteria come from the caller rather than makeCriteria: those
// solves stop against a baseline computed once from the ORIGINAL system (a warm
// start's residual is not a fair yardstick for itself), not buildKrylov's
// per-generate() rhs_norm. `what` names the caller in the "unknown solver" message.
std::shared_ptr<gko::LinOp> generateBasicSolver(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>& criteria,
    const char* what
);

// Fill the SolveResult every solve entry point returns (krylov/result.hpp for the
// key-set contract). `diagnostic` is left empty here and only the stationary V-cycle
// fills it: its thresholds are calibrated for a V-cycle's roughly constant
// contraction and say nothing useful about a Krylov method, whose rate varies.
inline SolveResult makeSolveResult(
    std::int64_t num_iters, double res_norm, bool converged, const std::vector<double>& res_history
)
{
    SolveResult r;
    r.num_iters = num_iters;
    r.res_norm = res_norm;
    r.converged = converged;
    r.res_history = res_history;
    // Geometric mean of the per-iteration residual reduction; the history holds the
    // initial residual plus one entry per iteration, hence size() - 1 reductions.
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

// Overload for the common case: counters from a gko::log::Convergence logger, history
// from a ResidualHistoryLogger, both attached via add_logger before apply().
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
