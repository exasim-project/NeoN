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

// Build the (Iteration, ResidualNorm[, ResidualNorm]) stopping-criteria chain
// shared by every Krylov solve in this file: always stop on `max_iter`
// iterations or a ResidualNorm criterion with the given `baseline` (e.g.
// gko::stop::mode::rhs_norm for a relative rtol against ||rhs||, or
// gko::stop::mode::absolute when the caller has already folded rtol into an
// absolute `reduction_factor`); when `atol > 0.0` a second, absolute-baseline
// ResidualNorm criterion is appended (the second baseline is always
// gko::stop::mode::absolute — every call site wants the plain ||r|| <= atol
// test, so there is nothing left for a caller to override). Passing
// atol = 0.0 reproduces call sites that never had an atol branch at all, so
// this single helper covers all of them exactly.
//
// `norm` selects which norm the residual criteria measure: "l2" (Ginkgo's
// gko::stop::ResidualNorm, the default and the historical behaviour) or "linf"
// (ResidualNormInf, MLMG's norm — see stopNormInf.hpp). The iteration
// criterion is norm-independent.
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

// Build a Krylov solver over `op`, stopping on iteration count, the relative
// residual ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() is
// reused across right-hand sides), or — when atol > 0 — the absolute residual
// ||r|| <= atol. A non-null `precond` (an already-generated LinOp, e.g.
// MlmgPrecond) is attached as the solver's generated preconditioner. `norm`
// picks the norm both residual tests measure in ("l2" | "linf").
//
// `precond`/`norm` have no defaults: KrylovSolver::build (their one
// caller) always passes both explicitly, so a default here would be dead.
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
// oneshot.cpp. Unlike buildKrylov, the criteria are supplied by the caller
// rather than built here: those solves stop on a baseline computed once from
// the ORIGINAL system (a warm start's residual is not a fair yardstick for
// itself), not buildKrylov's per-generate() rhs_norm, so the two cannot share
// a criteria-building step. `what` names the caller in the "unknown solver"
// message — ginkgo_solve_composite and ginkgo_solve_face_coeffs are the only
// two callers, and carried byte-identical dispatch chains (differing only in
// that name) before this consolidation.
std::shared_ptr<gko::LinOp> generateBasicSolver(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>& criteria,
    const char* what
);

// Assemble the {num_iters, res_norm, converged, res_history, contraction,
// diagnostic} result dict returned by every solve entry point (the epilogue
// duplicated at several ginkgoSolve.cpp call sites). `res_history` is built
// from `resLogger.history()`.
//
// `contraction` and `diagnostic` are filled for EVERY path, not just the
// stationary V-cycle that needs them, because the key set is a contract: a
// caller reads one dict without branching on which solver produced it (see
// test_gmg_solver_stats_keys_match_cg). `diagnostic` is left empty here and
// only the stationary path fills it in -- its thresholds are calibrated for a
// V-cycle's roughly constant contraction and say nothing useful about a Krylov
// method, whose rate varies over the run.
inline SolveResult makeSolveResult(
    std::int64_t num_iters, double res_norm, bool converged, const std::vector<double>& res_history
)
{
    SolveResult r;
    r.num_iters = num_iters;
    r.res_norm = res_norm;
    r.converged = converged;
    r.res_history = res_history;
    // Geometric mean of the per-iteration residual reduction. The history holds
    // the initial residual plus one entry per iteration, so the number of
    // reductions is size() - 1.
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

// Overload for the common case: num_iters/converged come from a
// gko::log::Convergence logger and res_history from a ResidualHistoryLogger,
// both attached to the solver via add_logger before apply().
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
