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

// MLMG's stopping norm, which Ginkgo lacks: max|r_i| reduced with amrex::Reduce over
// Ginkgo's own device pointer (Dense offers compute_norm2/compute_norm1 only). Why the
// norm is a choice, and the measured comparison:
// report/blockamr-linear-algebra-notes.md#norms

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

// max |v_i| over a single-column Dense: on its own executor (hence the synchronize --
// AMReX's stream is unordered against the one Ginkgo wrote `v` on) and Max-reduced across
// ranks, or a rank stops on its own residual. Size and values come from the local
// accessors: get_size() on a distributed vector is the GLOBAL row count.
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

// ||r||_inf <= tau * baseline. Single right-hand side only: a per-column inf-norm would
// need a per-column reduction amrex::Reduce cannot express over a flat pointer.
class ResidualNormInf : public gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>
{
    friend class gko::EnablePolymorphicObject<ResidualNormInf, gko::stop::Criterion>;

public:

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        // As in gko::stop::ResidualNorm: the factor the baseline is multiplied by.
        double GKO_FACTORY_PARAMETER_SCALAR(reduction_factor, 1e-10);

        // absolute | rhs_norm | initial_resnorm, as gko::stop::ResidualNorm but in the
        // inf-norm. MLMG uses the GREATER of the two, which coincide at a zero guess.
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
            // Not a missing residual: Ir asks for the cheap checks first and calls again
            // with it. Throwing here instead made ResidualNormInf unusable with ir/mpir.
            return false;
        }
        if (r == nullptr)
        {
            // A solver publishing only the implicit squared 2-norm cannot be stopped on
            // an inf-norm; refuse rather than silently stop on a different criterion.
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
            // copy_from into the existing storage: the solver holds pointers into it.
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
        // A zero baseline would make the target 0 and the solve unstoppable.
        return (nrm > 0.0) ? nrm : 1.0;
    }

    double target_ = 0.0;
    double last_ = 0.0;
};

} // namespace blockamr::la
