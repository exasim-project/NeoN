// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#if NF_WITH_GINKGO
#include "NeoN/linearAlgebra/distributedGinkgoSolver.hpp"
#endif
#endif


namespace NeoN::dsl
{

namespace detail
{
template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    la::LinearSystem<
        typename VectorType::ElementType,
        la::CSRMatrix<typename VectorType::ElementType, IndexType>>& ls,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<PostAssemblyBase<typename VectorType::ElementType, IndexType>> ps
)
{
    exp.read(fvSchemes);
    exp.assemble(t, dt, ls, ps);

    // TODO move that to expression explicit operation or
    // into functor ?
    // subtract the explicit source term from the rhs
    auto expTmp = exp.explicitOperation(solution.mesh().nCells());
    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

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
    std::vector<PostAssemblyBase<typename VectorType::ElementType, IndexType>> ps
)
{
    auto& mesh = solution.mesh();

    if (!mesh.isDistributed())
    {
        // Serial path (unchanged)
        auto [sparsity, ls] = exp.assemble(mesh, t, dt, ps);

        auto expTmp = exp.explicitOperation(mesh.nCells());
        auto [vol, expSource, rhs] = views(mesh.cellVolumes(), expTmp, ls.rhs());
        parallelFor(
            solution.exec(),
            {0, rhs.size()},
            NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
        );

        auto solver = la::Solver(solution.exec(), fvSolution);
        fence(solution.exec());
        return solver.solve(ls, solution.internalVector());
    }

#ifdef NF_WITH_MPI_SUPPORT
    // Distributed outer iteration
    int maxOuterIters = 50;
    scalar outerTol = 1e-6;
    if (fvSolution.contains("outerIterations"))
    {
        maxOuterIters = fvSolution.get<int>("outerIterations");
    }
    if (fvSolution.contains("outerTolerance"))
    {
        outerTol = fvSolution.get<scalar>("outerTolerance");
    }

    auto& comm = *mesh.communicator();

    // Initial assembly to get sparsity pattern
    auto [sparsity, ls] = exp.assemble(mesh, t, dt, ps);

    la::SolverStats lastStats;
    for (int outerIter = 0; outerIter < maxOuterIters; ++outerIter)
    {
        // 1. Sync ghost cells
        comm.startComm(solution.internalVector(), "outerSolve");
        comm.finaliseComm(solution.internalVector(), "outerSolve");

        // 2. Update proc-boundary BCs with new ghost values
        solution.correctBoundaryConditions();

        // 3. Re-assemble with updated ghost values
        ls.reset();
        exp.assemble(t, dt, ls, ps);

        // 4. Subtract explicit source term from RHS
        auto expTmp = exp.explicitOperation(mesh.nCells());
        auto [vol, expSource, rhs] = views(mesh.cellVolumes(), expTmp, ls.rhs());
        parallelFor(
            solution.exec(),
            {0, rhs.size()},
            NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
        );

        // 5. Solve distributed system
        fence(solution.exec());
#if NF_WITH_GINKGO
        if constexpr (std::is_same_v<typename VectorType::ElementType, scalar>)
        {
            la::DistributedGinkgoSolver distSolver(solution.exec(), fvSolution, mesh);
            lastStats = distSolver.solve(ls, solution.internalVector());
        }
        else
#endif
        {
            la::Solver localSolver(solution.exec(), fvSolution);
            lastStats = localSolver.solve(ls, solution.internalVector());
        }

        // 6. Global convergence check
        scalar localRes = lastStats.entries.back().finalResNorm;
        scalar globalRes = 0.0;
        MPI_Allreduce(&localRes, &globalRes, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (globalRes < outerTol) break;
    }
    return lastStats;
#else
    NF_ERROR_EXIT("Distributed solve requires MPI support");
    return {};
#endif
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
    std::vector<PostAssemblyBase<typename VectorType::ElementType, IndexType>> p = {}
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
