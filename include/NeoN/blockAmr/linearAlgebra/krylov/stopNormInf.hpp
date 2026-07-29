// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/types.hpp"

// The convergence norm, as a choice: Ginkgo's criteria measure the residual in the
// 2-norm, AMReX's MLMG in the INFINITY norm relative to max(||b||_inf, ||r0||_inf)
// (AMReX_MLMG.H: MLResNormInf / MLRhsNormInf, MLMGNormType::greater,
// res_target = max(atol, max(rtol,1e-16) * max_norm)). Two solvers stopping on
// different norms answer different questions, so their iteration counts are not
// directly comparable -- which matters because the interesting comparisons are close:
// mlmg at 9 iterations against mf-gmgk at 10. Which criterion is stricter follows from
// each vector's max/rms ratio C,
//   ||r||_inf / ||b||_inf = (C_r / C_b) * ||r||_2 / ||b||_2,
// and neither dominates a priori; the point is only that a comparison should be able to
// hold the norm fixed.
//
// Ginkgo has no inf-norm reduction (Dense offers compute_norm2/compute_norm1 only), so
// the criterion below reduces max|r_i| itself with amrex::Reduce over Ginkgo's own
// device pointer -- one cross-runtime synchronisation per iteration, the same one
// MLMG's ResNormInf pays.

namespace blockamr::la
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

// max |v_i| over a single-column Dense, on its own executor and reduced across ranks.
//
// The AMReX reduction runs on the AMReX stream, unordered against the one Ginkgo wrote
// `v` on -- hence the explicit synchronize, the same guard the GMG preconditioners use
// before reading Ginkgo-written data. The cross-rank Max mirrors MultiFab::norminf and
// the native V-cycle's norms (gmgKernels.hpp reduceResidNorms): every reduction the
// stopping test consults has to be global, or a rank stops on its own residual. It is
// safe as a collective because the criterion is driven by a Ginkgo solver whose
// iteration sequence is identical on every rank. Size and values come from the local
// accessors -- get_size() on a distributed vector is the GLOBAL row count, and reading
// that many values off the local buffer would run off the end of it.
inline double normInf(const gko::LinOp* v)
{
    const auto n = localRows(v);
    double m = 0.0;
    if (n > 0)
    {
        const double* p = localValues<double>(v);
        auto exec = v->get_executor();
        if (exec->get_master().get() == exec.get())
        {
            for (gko::size_type i = 0; i < n; ++i)
            {
                m = std::max(m, std::abs(p[i]));
            }
        }
        else
        {
            exec->synchronize();
            // Max and Min rather than a Max over |p_i|: the pointer overloads take no
            // device lambda, so this header stays includable anywhere.
            const auto hi = amrex::Reduce::Max<double>(n, p);
            const auto lo = amrex::Reduce::Min<double>(n, p);
            m = std::max(std::abs(hi), std::abs(lo));
        }
    }
    amrex::ParallelAllReduce::Max(m, amrex::ParallelContext::CommunicatorSub());
    return m;
}

// ||r||_inf <= tau * baseline, in MLMG's norm. Single right-hand side only: the stack
// solves one system at a time, and a per-column inf-norm would need a per-column
// reduction amrex::Reduce cannot express over a flat pointer.
class ResidualNormInf : public gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>
{
    friend class gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>;

public:

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // As in gko::stop::ResidualNorm: the factor the baseline is multiplied by.
        double GKO_FACTORY_PARAMETER_SCALAR(reduction_factor, 1e-10);

        // absolute | rhs_norm | initial_resnorm, as for gko::stop::ResidualNorm but
        // in the inf-norm. MLMG uses the GREATER of rhs_norm and initial_resnorm,
        // which coincide at a zero initial guess; a warm start's baseline is the
        // caller's choice.
        gko::stop::mode GKO_FACTORY_PARAMETER_SCALAR(baseline, gko::stop::mode::rhs_norm);
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNormInf, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

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
        const gko::LinOp* r = updater.residual_;
        if (r == nullptr && updater.ignore_residual_check_)
        {
            // Not a missing residual: a solver saying "do the cheap checks now, I
            // will call you again with the residual". Ir does this past the first
            // iteration (core/solver/update_residual.hpp), running the
            // iteration-count criterion before forming b - A x so a solver that has
            // already stopped never pays for the residual. Throwing here instead of
            // returning false made ResidualNormInf unusable with ir/mpir.
            return false;
        }
        if (r == nullptr)
        {
            // The genuine case: a solver publishing only the implicit squared 2-norm
            // cannot be stopped on an inf-norm at all, so refuse rather than silently
            // stop on a different criterion than the caller asked for.
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
            // copy_from into the existing storage: the solver keeps pointers into this
            // array, so it must not be reseated.
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
        if (v == nullptr)
        {
            throw gko::NotSupported(
                __FILE__, __LINE__, __func__, std::string("ResidualNormInf needs ") + what
            );
        }
        const double nrm = normInf(v);
        // A zero baseline would make the target 0 and the solve unstoppable
        // (gko::stop::ResidualNorm has the same degeneracy).
        return (nrm > 0.0) ? nrm : 1.0;
    }

    double target_ = 0.0;
    double last_ = 0.0;
};

} // namespace blockamr::la
