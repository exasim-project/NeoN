// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la::ginkgo
{

gko::config::pnode parse(const Dictionary& dict);

class GinkgoSolver : public SolverFactory::template Register<GinkgoSolver>
{

    using Base = SolverFactory::template Register<GinkgoSolver>;

public:

    GinkgoSolver(Executor exec, const Dictionary& solverConfig)
        : Base(exec), gkoExec_(getGkoExecutor(exec)), config_(parse(solverConfig)),
          factory_(gko::config::parse(
                       config_, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
          )
                       .on(gkoExec_))
    {}

    static std::string name() { return "Ginkgo"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual SolverStats
    solve(const LinearSystem<scalar, localIdx>& sys, Vector<scalar>& x) const final
    {
        using vec = gko::matrix::Dense<scalar>;

        auto retrieve = [](const auto& in)
        {
            auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
            scalar res = host->copy_from(in)->at(0);
            return res;
        };

        auto nrows = sys.rhs().size();

        auto gkoMtx = detail::createGkoMtx(gkoExec_, sys);
        auto solver = factory_->generate(gkoMtx);

        std::shared_ptr<const gko::log::Convergence<scalar>> logger =
            gko::log::Convergence<scalar>::create();
        solver->add_logger(logger);

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto rhs2 = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);

        auto one = gko::initialize<vec>({1.0}, gkoExec_);
        auto neg_one = gko::initialize<vec>({-1.0}, gkoExec_);
        auto init = gko::initialize<vec>({0.0}, gkoExec_);
        gkoMtx->apply(one, gkoX, neg_one, rhs2);
        rhs->compute_norm2(init);
        scalar initResNorm = retrieve(init);

        solver->apply(rhs, gkoX);

        scalar finalResNorm = retrieve(gko::as<vec>(logger->get_residual_norm()));

        auto numIter = label(logger->get_num_iterations());

        return {numIter, initResNorm, finalResNorm};
    }

    virtual std::unique_ptr<SolverFactory> clone() const final
    {
        // FIXME
        return {};
    }

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
};


}

#endif
