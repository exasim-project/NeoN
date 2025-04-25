// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <concepts>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"

// FIXME
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"


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
    if (exp.size() == 0)
    {
        NF_ERROR_EXIT("Empty expression, no temporal or implicit terms to solve.");
    }

    if (exp.temporalOperators().size() > 1)
    {
        NF_ERROR_EXIT("Found more then one temporal operator ");
    }

    exp.read(fvSchemes);
    if (exp.temporalOperators().size() == 1)
    {
        // TODO Make sure that scheme is explicit
        // integrate equations in time
        timeIntegration::TimeIntegration<VectorType> timeIntegrator(
            fvSchemes.subDict("ddtSchemes"), fvSolution, exp.temporalOperators()[0].getType()
        );
        timeIntegrator.solve(exp, solution, t, dt);
        return;
    }

    // solve sparse matrix system
    using ValueType = typename VectorType::ElementType;

    // TODO store the sparsity pattern to avoid recomputing it every time
    auto sparsity = NeoN::finiteVolume::cellCentred::SparsityPattern(solution.mesh());
    auto ls = la::createEmptyLinearSystem<
        ValueType,
        localIdx,
        NeoN::finiteVolume::cellCentred::SparsityPattern>(sparsity);

    exp.assembleLinearSystem(ls);
    auto expTmp = exp.explicitOperation(solution.mesh().nCells());

    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());

    // subtract the explicit source term from the rhs
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

    auto solver = NeoN::la::Solver(solution.exec(), fvSolution);
    solver.solve(ls, solution.internalVector());
}

} // namespace dsl
