// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
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


namespace NeoN::dsl
{

namespace detail
{
template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    la::LinearSystem<typename VectorType::ElementType>& ls,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> ps = {}
)
{
    exp.read(fvSchemes);
    exp.assemble(t, dt, ls, solution.mesh(), ps);

    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());

    // Do some sanity checks before trying to solve
    NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
    return solver.solve(ls, solution.internalVector());
}

template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> ps = {}
)
{
    auto ls = exp.assemble(solution.mesh(), t, dt, ps);

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
template<typename VectorType, typename IndexType>
la::SolverStats solve(
    Expression<typename VectorType::ElementType, IndexType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> p = {}
)
{
    if (exp.temporalOperators().size() == 0 && exp.spatialOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    exp.read(fvSchemes);
    auto integrator = timeIntegration::TimeIntegration<VectorType>(
        fvSchemes.subDict("timeIntegration"), fvSolution
    );

    if (exp.temporalOperators().size() > 0 && integrator.explicitIntegration())
    {
        // integrate equations in time
        integrator.solve(exp, solution, t, dt);
        return {{.numIter = -1, .initResNorm = 0, .finalResNorm = 0, .solveTime = 0}};
    }
    else
    {
        return detail::iterativeSolveImpl(exp, solution, t, dt, fvSolution, p);
    }
}

} // namespace dsl
