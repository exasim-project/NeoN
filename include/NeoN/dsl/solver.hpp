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

/* @brief solve an expression
 *
 * @param exp - Expression which is to be solved/updated.
 * @param solution - Solution field, where the solution will be 'written to'.
 * @param t - the time at the start of the time step.
 * @param dt - time step for the temporal integration
 * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
 * @param fvSolution - Dictionary containing linear solver properties
 */
template<typename VectorType>
void solve(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution
)
{
    // TODO:
    if (exp.temporalOperators().size() == 0 && exp.spatialOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    exp.read(fvSchemes);
    if (exp.temporalOperators().size() > 0)
    {
        // integrate equations in time
        timeIntegration::TimeIntegration<VectorType> timeIntegrator(
            fvSchemes.subDict("ddtSchemes"), fvSolution
        );
        timeIntegrator.solve(exp, solution, t, dt);
    }
    else
    {
        // solve sparse matrix system
        using ValueType = typename VectorType::ElementType;

        auto sparsity = la::SparsityPattern(solution.mesh());
        auto ls = la::createEmptyLinearSystem<ValueType, localIdx>(solution.mesh(), sparsity);

        exp.implicitOperation(ls);
        auto expTmp = exp.explicitOperation(solution.mesh().nCells());

        auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());

        // subtract the explicit source term from the rhs
        parallelFor(
            solution.exec(),
            {0, rhs.size()},
            KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
        );

        auto solver = la::Solver(solution.exec(), fvSolution);
        solver.solve(ls, solution.internalVector());
    }
}

} // namespace dsl
