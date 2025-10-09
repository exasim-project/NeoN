// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <concepts>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"


namespace NeoN::dsl
{

namespace detail
{

// TODO VectorType not needed use ValueType
template<typename VectorType>
la::SolverStats iterative_solve_impl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<OpFunctor<typename VectorType::ElementType>> ps
)
{
    using ValueType = typename VectorType::ElementType;
    auto [sparsity, ls] = exp.assemble(solution.mesh(), t, dt, ps);

    // TODO move that to expression explicit operation or
    // into functor ?
    // subtract the explicit source term from the rhs
    auto expTmp = exp.explicitOperation(solution.mesh().nCells());
    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    return solver.solve(ls, solution.internalVector());
}
}

/* @brief solve an expression
 *
 * @param exp - Expression which is to be solved/updated.
 * @param solution - Solution field, where the solution will be 'written to'.
 * @param t - the time at the start of the time step.
 * @param dt - time step for the temporal integration
 * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
 * @param fvSolution - Dictionary containing linear solver properties
 * @param p - A chainable functor that performs manipulations on the assembled system
 */
template<typename VectorType>
la::SolverStats solve(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<OpFunctor<typename VectorType::ElementType>> p = {}
)
{
    if (exp.temporalOperators().size() == 0 && exp.spatialOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    exp.read(fvSchemes);
    auto integrator =
        timeIntegration::TimeIntegration<VectorType>(fvSchemes.subDict("ddtSchemes"), fvSolution);

    if (exp.temporalOperators().size() > 0 && integrator.direct())
    {
        // integrate equations in time
        integrator.solve(exp, solution, t, dt);
        return {.numIter = -1, .initResNorm = 0, .finalResNorm = 0, .solveTime = 0};
    }
    else
    {
        return detail::iterative_solve_impl(exp, solution, t, dt, fvSchemes, fvSolution, p);
    }
}

} // namespace dsl
