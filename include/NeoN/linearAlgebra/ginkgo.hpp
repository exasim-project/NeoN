// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la::ginkgo
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec);

namespace detail
{

template<typename T>
gko::array<T> createGkoArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

// template<typename T>
// gko::detail::const_array_view<T>
// createConstGkoArray(std::shared_ptr<const gko::Executor> exec, const std::span<const T> values)
// {
//     return gko::make_const_array_view(exec, values.size(), values.data());
// }

template<typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec, const LinearSystem<ValueType, IndexType>& sys
)
{
    auto nrows = static_cast<gko::dim<2>::dimension_type>(sys.rhs().size());
    auto mtx = sys.view().matrix;
    // NOTE we get a const view of the system but need a non const view to vals and indices
    // auto vals = createConstGkoArray(exec, mtx.values).copy_to_array();
    auto vals = gko::array<ValueType>::view(
        exec,
        static_cast<gko::size_type>(mtx.values.size()),
        const_cast<ValueType*>(mtx.values.data())
    );
    // auto col = createGkoArray(exec, mtx.colIdxs);
    auto col = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.colIdxs.size()),
        const_cast<IndexType*>(mtx.colIdxs.data())
    );
    // auto row = createGkoArray(exec, mtx.rowOffs);
    auto row = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.rowOffs.size()),
        const_cast<IndexType*>(mtx.rowOffs.data())
    );
    return gko::share(gko::matrix::Csr<ValueType, IndexType>::create(
        exec, gko::dim<2> {nrows, nrows}, vals, col, row
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, ValueType* ptr, localIdx s)
{
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, createGkoArray(exec, std::span {ptr, size}), 1
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, const ValueType* ptr, localIdx s)
{
    auto size = static_cast<std::size_t>(s);
    auto const_array_view = gko::array<ValueType>::const_view(exec, size, ptr);
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, const_array_view.copy_to_array(), 1
    ));
}

}
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
