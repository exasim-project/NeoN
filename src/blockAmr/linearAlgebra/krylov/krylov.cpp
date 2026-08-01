// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylov.hpp"

#include "NeoN/linearAlgebra/ginkgo.hpp"

#include <stdexcept>

namespace blockamr::la
{

namespace
{

using Criteria = std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>;

// Attach `precond` as the generated preconditioner, shared by all five solver builders.
// Returns `params` itself: GCC 13 rejects the unique_ptr<SolverType> -> shared_ptr<gko::LinOp>
// conversion inside a function template, so it has to stay in a non-template body.
template<class Params>
Params withPrecond(Params params, const std::shared_ptr<const gko::LinOp>& precond)
{
    if (precond)
    {
        params.with_generated_preconditioner(precond);
    }
    return params;
}

// The five methods that take `precond` as a preconditioner. Non-template for the reason
// withPrecond gives above.
std::shared_ptr<gko::LinOp> generatePreconditionedKrylov(
    const std::string& solver,
    const std::shared_ptr<const gko::Executor>& exec,
    const std::shared_ptr<const gko::LinOp>& op,
    const Criteria& criteria,
    const std::shared_ptr<const gko::LinOp>& precond
)
{
    if (solver == "cg")
    {
        return withPrecond(gko::solver::Cg<double>::build().with_criteria(criteria), precond)
            .on(exec)
            ->generate(op);
    }
    if (solver == "bicgstab")
    {
        return withPrecond(gko::solver::Bicgstab<double>::build().with_criteria(criteria), precond)
            .on(exec)
            ->generate(op);
    }
    if (solver == "gmres")
    {
        return withPrecond(gko::solver::Gmres<double>::build().with_criteria(criteria), precond)
            .on(exec)
            ->generate(op);
    }
    // FLEXIBLE outer solvers: they do not assume a fixed preconditioner, so they suit a V-cycle
    // whose bottom is an adaptive Krylov solve (loose gmg_bottom_rtol). Both cost more per
    // iteration than Cg, so they are opt-in; Fcg still needs a SYMMETRIC operator, Gcr does not.
    if (solver == "gcr")
    {
        return withPrecond(gko::solver::Gcr<double>::build().with_criteria(criteria), precond)
            .on(exec)
            ->generate(op);
    }
    if (solver == "fcg")
    {
        return withPrecond(gko::solver::Fcg<double>::build().with_criteria(criteria), precond)
            .on(exec)
            ->generate(op);
    }
    throw std::runtime_error("ginkgo: unknown solver '" + solver + "'");
}

// Iterative refinement x <- x + relax * S(b - A x) with S = `precond`, the generated GMG
// V-cycle; relaxation 1.0 is plain Richardson, and default_initial_guess `provided` lets
// the incoming x seed it (warm start). Inner solver via with_generated_solver.
std::shared_ptr<gko::LinOp> generateRefinement(
    const std::shared_ptr<const gko::Executor>& exec,
    const std::shared_ptr<const gko::LinOp>& op,
    const Criteria& criteria,
    const std::shared_ptr<const gko::LinOp>& precond
)
{
    auto params = gko::solver::Ir<double>::build().with_criteria(criteria);
    params.with_relaxation_factor(1.0);
    if (precond)
    {
        params.with_generated_solver(precond);
    }
    return params.on(exec)->generate(op);
}

} // namespace

std::shared_ptr<const gko::Executor> makeExecutor(const NeoN::Executor& executor)
{
    // No mapping of our own: NeoN owns it (memoization, the Kokkos finalize hook, the stream).
    return NeoN::la::ginkgo::getGkoExecutor(executor);
}

std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const StopSpec& stop,
    std::shared_ptr<const gko::LinOp> precond
)
{
    auto criteria = makeCriteria(exec, stop);
    // "ir" is the odd one out: `precond` is its inner SOLVER, not a preconditioner.
    if (solver == "ir")
    {
        return generateRefinement(exec, op, criteria, precond);
    }
    return generatePreconditionedKrylov(solver, exec, op, criteria, precond);
}

std::shared_ptr<gko::LinOp> generateBasicSolver(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    const std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>& criteria,
    const char* what
)
{
    if (solver == "cg")
    {
        return gko::solver::Cg<double>::build().with_criteria(criteria).on(exec)->generate(op);
    }
    if (solver == "bicgstab")
    {
        return gko::solver::Bicgstab<double>::build().with_criteria(criteria).on(exec)->generate(op
        );
    }
    if (solver == "gmres")
    {
        return gko::solver::Gmres<double>::build().with_criteria(criteria).on(exec)->generate(op);
    }
    throw std::runtime_error(std::string(what) + ": unknown solver '" + solver + "'");
}

} // namespace blockamr::la
