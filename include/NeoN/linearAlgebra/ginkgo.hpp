// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NF_WITH_GINKGO

#include <chrono>

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>
#include <ginkgo/extensions/config/json_config.hpp>

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
    solve(const LinearSystem<scalar, localIdx>& sys, Vector<scalar>& x) const final;

    virtual SolverStats solve(const LinearSystem<Vec3, localIdx>& sys, Vector<Vec3>& x) const final;

    // TODO why use a smart pointer here?
    virtual std::unique_ptr<SolverFactory> clone() const final
    {
        NF_ERROR_EXIT("Not implemented");
        return {};
    }

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
};


}

#endif
