// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "executor.hpp"
#include "krylov.hpp"

#include "NeoN/linearAlgebra/ginkgo.hpp"

#include <stdexcept>

namespace blockamr::solvers
{

std::shared_ptr<const gko::Executor> makeExecutor(const NeoN::Executor& executor)
{
    // No mapping of our own: NeoN already owns this one (memoization, the Kokkos
    // finalize hook, and the execution-space stream). Restating it here is how the
    // two copies drift apart.
    return NeoN::la::ginkgo::getGkoExecutor(executor);
}

std::shared_ptr<gko::LinOp> buildKrylov(
    const std::string& solver,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol,
    double atol,
    std::shared_ptr<const gko::LinOp> precond,
    const std::string& norm
)
{
    auto criteria = makeCriteria(
        exec, max_iter, gko::stop::mode::rhs_norm, rtol, atol, gko::stop::mode::absolute, norm
    );
    if (solver == "cg")
    {
        auto params = gko::solver::Cg<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "bicgstab")
    {
        auto params = gko::solver::Bicgstab<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "gmres")
    {
        auto params = gko::solver::Gmres<double>::build().with_criteria(criteria);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "gcr" || solver == "fcg")
    {
        // FLEXIBLE outer solvers: unlike Cg and Bicgstab they do not assume the
        // preconditioner is the same linear operator on every apply, so they are
        // the right outer method when the V-cycle's bottom is solved by an
        // adaptive Krylov method (gmg_bottom_solver != 'smoother' with a loose
        // gmg_bottom_rtol). Both cost more per iteration than Cg -- Gcr stores a
        // growing search space, Fcg one extra vector -- so they are opt-in
        // rather than the default; the stationary route (a tight bottom rtol,
        // keeping solver='cg') is usually cheaper. Fcg still needs a SYMMETRIC
        // operator; Gcr does not.
        auto criteriaFlex = criteria;
        if (solver == "gcr")
        {
            auto params = gko::solver::Gcr<double>::build().with_criteria(criteriaFlex);
            if (precond)
            {
                params.with_generated_preconditioner(precond);
            }
            return params.on(exec)->generate(op);
        }
        auto params = gko::solver::Fcg<double>::build().with_criteria(criteriaFlex);
        if (precond)
        {
            params.with_generated_preconditioner(precond);
        }
        return params.on(exec)->generate(op);
    }
    if (solver == "ir")
    {
        // Iterative refinement x <- x + relax * S(b - A x), where S is the
        // already-generated inner solver `precond` (the GMG V-cycle LinOp). With
        // relaxation_factor 1.0 this is plain Richardson driven by the V-cycle,
        // Ginkgo's idiomatic counterpart of the native solver="gmg" loop.
        // default_initial_guess defaults to `provided`, so the incoming x seeds
        // the iteration (the persistent-solver warm-start contract).
        auto params = gko::solver::Ir<double>::build().with_criteria(criteria);
        params.with_relaxation_factor(1.0);
        if (precond)
        {
            params.with_generated_solver(precond);
        }
        return params.on(exec)->generate(op);
    }
    throw std::runtime_error("ginkgo: unknown solver '" + solver + "'");
}

} // namespace blockamr::solvers
