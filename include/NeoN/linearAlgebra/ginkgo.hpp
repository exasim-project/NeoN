// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/solver.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/utilities.hpp"


namespace NeoFOAM::la::ginkgo
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

    virtual void solve(const LinearSystem<scalar, localIdx>& sys, Field<scalar>& x) const final
    {
        size_t nrows = sys.rhs().size();

        auto gkoMtx = detail::createGkoMtx(gkoExec_, sys);
        auto solver = factory_->generate(gkoMtx);
        // std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        //     gko::log::Convergence<ValueType>::create();
        // solver->add_logger(logger);

        auto rhs = detail::createGkoDense(gkoExec_, sys.rhs().data(), nrows);
        auto gkoX = detail::createGkoDense(gkoExec_, x.data(), nrows);

        solver->apply(rhs, gkoX);
        // auto res_norm = gko::as<gko::matrix::Dense<ValueType>>(logger->get_residual_norm());
        // auto res_norm_host = gko::matrix::Dense<ValueType>::create(
        //     res_norm->get_executor()->get_master(), gko::dim<2> {1}
        // );
        // res_norm_host->copy_from(res_norm);
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
