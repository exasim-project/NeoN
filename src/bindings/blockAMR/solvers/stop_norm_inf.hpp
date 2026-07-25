// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "types.hpp"

// ---------------------------------------------------------------------------
// The convergence norm, as a choice.
//
// Ginkgo's stopping criteria measure the residual in the 2-norm; AMReX's MLMG
// measures it in the INFINITY norm, relative to max(||b||_inf, ||r0||_inf)
// (AMReX_MLMG.H: MLResNormInf / MLRhsNormInf, MLMGNormType::greater, and
// res_target = max(atol, max(rtol,1e-16) * max_norm)). Two solvers stopping on
// different norms are two solvers answering different questions, so an
// iteration count from one is not directly comparable with the other's -- which
// matters here precisely because the interesting comparisons are close: mlmg at
// 9 iterations against mf-gmgk at 10.
//
// The relation between the two is set by each vector's max/rms ratio:
//   ||r||_inf / ||b||_inf = (C_r / C_b) * ||r||_2 / ||b||_2,   C = max/rms
// so which criterion is the stricter one depends on how peaked the residual is
// against the right-hand side, and neither dominates a priori. The point of
// this header is not that one norm is better -- it is that a comparison should
// be able to hold the norm fixed.
//
// Ginkgo has no inf-norm reduction (Dense offers compute_norm2 and
// compute_norm1 only), so the criterion below reduces max|r_i| itself with
// amrex::Reduce over Ginkgo's own device pointer. That costs one cross-runtime
// synchronisation per iteration, the same one MLMG's ResNormInf pays.
// ---------------------------------------------------------------------------

namespace blockamr::solvers
{

// Which norm the stopping criterion (and the reported residual) measures.
enum class NormKind
{
    l2,
    linf
};

inline NormKind parseNorm(const std::string& norm)
{
    if (norm == "l2")
    {
        return NormKind::l2;
    }
    if (norm == "linf")
    {
        return NormKind::linf;
    }
    throw std::runtime_error(
        "ginkgo: unknown norm '" + norm + "' (expected 'l2' or 'linf'; 'linf' is MLMG's)"
    );
}

// max |v_i| over a single-column Dense, computed on its own executor.
//
// The AMReX reduction runs on the AMReX stream while Ginkgo wrote `v` on its
// own, and the two are unordered -- hence the explicit synchronize first, the
// same guard the GMG preconditioners use before reading Ginkgo-written data.
inline double normInf(const Dense* v)
{
    const auto n = v->get_size()[0] * v->get_size()[1];
    if (n == 0)
    {
        return 0.0;
    }
    const double* p = v->get_const_values();
    auto exec = v->get_executor();
    if (exec->get_master().get() == exec.get())
    {
        double m = 0.0;
        for (gko::size_type i = 0; i < n; ++i)
        {
            m = std::max(m, std::abs(p[i]));
        }
        return m;
    }
    exec->synchronize();
    // Max and Min rather than a Max over |p_i|: the pointer overloads take no
    // device lambda, so this header stays compilable wherever it is included.
    const auto hi = amrex::Reduce::Max<double>(n, p);
    const auto lo = amrex::Reduce::Min<double>(n, p);
    return std::max(std::abs(hi), std::abs(lo));
}

// ||r||_inf <= tau * baseline, in MLMG's norm.
//
// Single right-hand side only: the whole blockAMR stack solves one system at a
// time, and a per-column inf-norm would need a per-column reduction that
// amrex::Reduce cannot express over a flat pointer.
class ResidualNormInf : public gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>
{
    friend class gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>;

public:

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // Same meaning as gko::stop::ResidualNorm's: the factor the baseline is
        // multiplied by to get the target.
        double GKO_FACTORY_PARAMETER_SCALAR(reduction_factor, 1e-10);

        // absolute | rhs_norm | initial_resnorm, as for gko::stop::ResidualNorm
        // (measured in the inf-norm here). MLMG's default is the GREATER of
        // rhs_norm and initial_resnorm; with the zero initial guess these
        // coincide, and warm-started solves are the caller's choice of baseline.
        gko::stop::mode GKO_FACTORY_PARAMETER_SCALAR(baseline, gko::stop::mode::rhs_norm);
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNormInf, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    // The inf-norm of the residual at the last check (0 before the first one).
    [[nodiscard]] double last_norm() const { return last_; }

    // The absolute threshold this criterion resolved its baseline to.
    [[nodiscard]] double target() const { return target_; }

protected:

    // Required by Ginkgo's polymorphic-object machinery (create_default/clear).
    explicit ResidualNormInf(std::shared_ptr<const gko::Executor> exec)
        : gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>(std::move(exec))
    {}

    explicit ResidualNormInf(const Factory* factory, const gko::stop::CriterionArgs& args)
        : gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>(factory->get_executor(
        )),
          parameters_ {factory->get_parameters()}
    {
        const double tau = parameters_.reduction_factor;
        switch (parameters_.baseline)
        {
        case gko::stop::mode::absolute:
            target_ = tau;
            break;
        case gko::stop::mode::rhs_norm:
            target_ = tau * baselineNorm(args.b.get(), "b");
            break;
        case gko::stop::mode::initial_resnorm:
            target_ = tau * baselineNorm(args.initial_residual, "initial_residual");
            break;
        }
    }

    bool check_impl(
        gko::uint8 stoppingId,
        bool setFinalized,
        gko::array<gko::stopping_status>* stop_status,
        bool* one_changed,
        const Updater& updater
    ) override
    {
        const auto* r = dynamic_cast<const Dense*>(updater.residual_);
        if (r == nullptr)
        {
            // Ir and the Krylov solvers used here all publish the residual; a
            // solver that only publishes the implicit squared 2-norm cannot be
            // stopped on an inf-norm at all, so say so rather than fall back to
            // a different criterion than the caller asked for.
            throw gko::NotSupported(
                __FILE__, __LINE__, __func__, "ResidualNormInf needs the residual vector"
            );
        }
        last_ = normInf(r);
        const bool converged = last_ <= target_;
        if (converged && stop_status != nullptr)
        {
            auto exec = this->get_executor();
            const auto n = stop_status->get_size();
            gko::array<gko::stopping_status> host(exec->get_master(), *stop_status);
            for (gko::size_type i = 0; i < n; ++i)
            {
                host.get_data()[i].converge(stoppingId, setFinalized);
            }
            // copy_from into the existing storage: the solver holds this array
            // and keeps pointers into it, so it must not be reseated.
            exec->copy_from(exec->get_master(), n, host.get_const_data(), stop_status->get_data());
        }
        if (one_changed != nullptr)
        {
            *one_changed = converged;
        }
        return converged;
    }

private:

    static double baselineNorm(const gko::LinOp* v, const char* what)
    {
        const auto* d = dynamic_cast<const Dense*>(v);
        if (d == nullptr)
        {
            throw gko::NotSupported(
                __FILE__, __LINE__, __func__, std::string("ResidualNormInf needs ") + what
            );
        }
        const double nrm = normInf(d);
        // A zero baseline would make the target 0 and the solve unstoppable;
        // gko::stop::ResidualNorm has the same degeneracy, and the call sites
        // that precompute a baseline already fall back to the bare rtol.
        return (nrm > 0.0) ? nrm : 1.0;
    }

    double target_ = 0.0;
    double last_ = 0.0;
};

} // namespace blockamr::solvers
