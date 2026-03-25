// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
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

gko::config::pnode parse(const Dictionary& dict);

/** @brief create a ginkgo matrix by creating views */
template<typename NeoNMatrixType>
std::shared_ptr<const gko::LinOp> createGkoMtx(const NeoNMatrixType& mtx);

template<typename T>
gko::array<T> gkoArrayView(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

/** @brief Non-owning mutable Dense view; 1 column for scalar*, 3 columns for Vec3* */
template<typename T>
std::shared_ptr<gko::matrix::Dense<scalar>>
gkoVecView(std::shared_ptr<const gko::Executor> exec, T* ptr, localIdx s)
{
    constexpr std::size_t cols = std::is_same_v<T, Vec3> ? 3 : 1;
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<scalar>::create(
        exec,
        gko::dim<2> {size, cols},
        gkoArrayView<scalar>(exec, std::span<scalar> {reinterpret_cast<scalar*>(ptr), cols * size}),
        cols
    ));
}

/** @brief Non-owning const Dense view; 1 column for scalar*, 3 columns for Vec3* */
template<typename T>
std::shared_ptr<const gko::matrix::Dense<scalar>>
gkoVecView(std::shared_ptr<const gko::Executor> exec, const T* ptr, localIdx s)
{
    constexpr std::size_t cols = std::is_same_v<T, Vec3> ? 3 : 1;
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<scalar>::create_const(
        exec,
        gko::dim<2> {size, cols},
        gko::array<scalar>::const_view(exec, cols * size, reinterpret_cast<const scalar*>(ptr)),
        cols
    ));
}

/** @brief retrieve a scalar value from a ginkgo dense matrix on any executor */
template<typename InType>
scalar retrieve(const InType& in)
{
    using vec = gko::matrix::Dense<scalar>;
    auto host = vec::create(in->get_executor()->get_master(), gko::dim<2> {1});
    return host->copy_from(in)->at(0);
}

class GinkgoSolver : public SolverFactory::template Register<GinkgoSolver>
{

    using Base = SolverFactory::template Register<GinkgoSolver>;

public:

    GinkgoSolver(Executor exec, const Dictionary& solverConfig)
        : Base(exec), gkoExec_(getGkoExecutor(exec)), coupled_(solverConfig.get("coupled", false)),
          config_(parse(solverConfig)),
          factory_(gko::config::parse(
                       config_, gko::config::registry(), gko::config::make_type_descriptor<scalar>()
          )
                       .on(gkoExec_))
    {}

    static std::string name() { return "Ginkgo"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual SolverStats solveDist(
        const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
    ) const final;

    virtual SolverStats solve(
        const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
    ) const final;

    virtual SolverStats
    solve(const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x) const final;

    virtual SolverStats solve(
        const LinearSystem<scalar, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>, Vec3>&
            sys,
        Vector<Vec3>& x
    ) const final;

#ifdef NF_WITH_MPI_SUPPORT
    virtual SolverStats solveDist(
        const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& sys, Vector<scalar>& x
    ) const final;

    virtual SolverStats solveDist(
        const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& sys, Vector<Vec3>& x
    ) const final;
#endif

    // TODO why use a smart pointer here?
    virtual std::unique_ptr<SolverFactory> clone() const final
    {
        NF_ERROR_EXIT("Not implemented");
        return {};
    }

private:

    std::shared_ptr<const gko::Executor> gkoExec_;
    bool coupled_; // whether to solve LinearSystem<Vec3> as one or three systems
    gko::config::pnode config_;
    std::shared_ptr<const gko::LinOpFactory> factory_;
};


}

#endif
