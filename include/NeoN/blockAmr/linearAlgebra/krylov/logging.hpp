// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <cmath>
#include <vector>

#include "NeoN/blockAmr/core/types.hpp"

namespace blockamr::la
{

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

} // namespace blockamr::la
