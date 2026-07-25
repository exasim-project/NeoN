// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

#include <ginkgo/ginkgo.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stop_norm_inf.hpp"
#include "types.hpp"

namespace nb = nanobind;

namespace blockamr::solvers
{

// One long-lived CudaExecutor per process (see the note in ginkgo_solve): a
// per-call executor re-inits cuBLAS/cuSPARSE and disturbs AMReX's CUDA context
// at teardown. Assumes a single AMReX Initialize/Finalize cycle.
std::shared_ptr<const gko::Executor> makeExecutor(const std::string& executor);

// Per-iteration residual-norm history. Ginkgo's iteration_complete event
// hands (solver, b, x, it, residual, residual_norm, implicit_sq_norm, ...);
// the criteria used here make the solvers pass residual_norm = nullptr, so
// the norm is computed from the residual vector (with the implicit squared
// norm as a last resort). Scalars land on the solve executor, so device
// values are staged through the host master before reading.
class ResidualHistoryLogger : public gko::log::Logger
{
public:

    ResidualHistoryLogger() : gko::log::Logger(gko::log::Logger::iteration_complete_mask) {}

    void clear() { history_.clear(); }

    const std::vector<double>& history() const { return history_; }

protected:

    void on_iteration_complete(
        const gko::LinOp*,
        const gko::LinOp*,
        const gko::LinOp*,
        const gko::size_type&,
        const gko::LinOp* residual,
        const gko::LinOp* residual_norm,
        const gko::LinOp* implicit_sq_norm,
        const gko::array<gko::stopping_status>*,
        bool
    ) const override
    {
        if (auto norm = dynamic_cast<const Dense*>(residual_norm))
        {
            history_.push_back(readScalar(norm));
        }
        else if (auto res = dynamic_cast<const Dense*>(residual))
        {
            auto exec = res->get_executor();
            auto norm2 = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm2);
            history_.push_back(readScalar(norm2.get()));
        }
        else if (auto sq = dynamic_cast<const Dense*>(implicit_sq_norm))
        {
            history_.push_back(std::sqrt(std::abs(readScalar(sq))));
        }
    }

private:

    static double readScalar(const Dense* d)
    {
        auto exec = d->get_executor();
        if (exec->get_master().get() != exec.get())
        {
            auto host = gko::clone(exec->get_master(), d);
            return host->at(0, 0);
        }
        return d->at(0, 0);
    }

    mutable std::vector<double> history_;
};

// Build the (Iteration, ResidualNorm[, ResidualNorm]) stopping-criteria chain
// shared by every Krylov solve in this file: always stop on `max_iter`
// iterations or a ResidualNorm criterion with the given `baseline` (e.g.
// gko::stop::mode::rhs_norm for a relative rtol against ||rhs||, or
// gko::stop::mode::absolute when the caller has already folded rtol into an
// absolute `reduction_factor`); when `atol > 0.0` a second, absolute-baseline
// ResidualNorm criterion is appended (`atol_baseline` lets a caller override
// that second baseline, though every current call site uses the default).
// Passing atol = 0.0 reproduces call sites that never had an atol branch at
// all, so this single helper covers all of them exactly.
//
// `norm` selects which norm the residual criteria measure: "l2" (Ginkgo's
// gko::stop::ResidualNorm, the default and the historical behaviour) or "linf"
// (ResidualNormInf, MLMG's norm — see stop_norm_inf.hpp). The iteration
// criterion is norm-independent.
inline std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> makeCriteria(
    std::shared_ptr<const gko::Executor> exec,
    int max_iter,
    gko::stop::mode baseline,
    double reduction_factor,
    double atol,
    gko::stop::mode atol_baseline = gko::stop::mode::absolute,
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
        criteria.push_back(residual(atol_baseline, atol));
    }
    return criteria;
}

// Build a Krylov solver over `op`, stopping on iteration count, the relative
// residual ||r|| <= rtol*||rhs|| (recomputed per solve, so one generate() is
// reused across right-hand sides), or — when atol > 0 — the absolute residual
// ||r|| <= atol. A non-null `precond` (an already-generated LinOp, e.g.
// MlmgPrecond) is attached as the solver's generated preconditioner. `norm`
// picks the norm both residual tests measure in ("l2" | "linf").
std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol,
    double atol,
    std::shared_ptr<const gko::LinOp> precond = nullptr,
    const std::string& norm = "l2"
);

// Assemble the {num_iters, res_norm, converged, res_history} result dict
// returned by every Krylov solve entry point (the epilogue duplicated at
// several ginkgo_solve.cpp call sites). `res_history` is built from
// `resLogger.history()`.
inline nb::dict makeResultDict(
    std::int64_t num_iters, double res_norm, bool converged, const std::vector<double>& res_history
)
{
    nb::dict d;
    d["num_iters"] = num_iters;
    d["res_norm"] = res_norm;
    d["converged"] = converged;
    nb::list hist;
    for (double v : res_history)
    {
        hist.append(v);
    }
    d["res_history"] = hist;
    return d;
}

// Overload for the common case: num_iters/converged come from a
// gko::log::Convergence logger and res_history from a ResidualHistoryLogger,
// both attached to the solver via add_logger before apply().
inline nb::dict makeResultDict(
    const gko::log::Convergence<double>& logger,
    const ResidualHistoryLogger& resLogger,
    double res_norm
)
{
    return makeResultDict(
        static_cast<std::int64_t>(logger.get_num_iterations()),
        res_norm,
        logger.has_converged(),
        resLogger.history()
    );
}

} // namespace blockamr::solvers
