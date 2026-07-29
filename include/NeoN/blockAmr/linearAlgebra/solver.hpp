// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"
#include "NeoN/blockAmr/linearAlgebra/solve/persistent.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @class Solver
 * @brief Solves a LinearSystem. Holds an executor and a SolverConfig BY VALUE and
 *        nothing else -- no matrix, no rhs, no geometry.
 *
 * Deliberately NO factory (design §7.4): which Krylov method the config names is
 * decided once, in solverConfig.hpp's parseSolverKind.
 *
 * Returns `la::SolveResult`, as every solve entry point in this component does,
 * rather than the design's separate `SolverStats`: its key set is a contract
 * pinned by test_gmg_solver_stats_keys_match_cg, and a second near-identical
 * struct is the duplication this refactor removes.
 */
class Solver
{
public:

    Solver(const NeoN::Executor& exec, const la::SolverConfig& cfg) : exec_(exec), cfg_(cfg) {}

    // rhs comes from the system, not the caller (cf. NeoN::la::Solver::solve(ls,
    // field)). `sol`'s incoming values seed the initial guess, as elsewhere here.
    la::SolveResult solve(const LinearSystem& system, amrex::MultiFab& sol) const
    {
        // Refused: these three want the hierarchy as the SOLVER (a stationary
        // V-cycle loop, or Ginkgo's Ir with the cycle as inner solver) -- a
        // different object with a different stopping test than the Krylov solver
        // this class builds. Checked BEFORE building the preconditioner, so the
        // refusal does not first pay for a hierarchy.
        if (cfg_.solverKind == la::SolverKind::gmg || cfg_.solverKind == la::SolverKind::ir
            || cfg_.solverKind == la::SolverKind::mpir)
        {
            throw std::runtime_error(
                "la::Solver: solver '" + cfg_.solver
                + "' needs the GMG hierarchy, which is built from the coefficient fields rather "
                  "than from a LinOp; use a Krylov method"
            );
        }
        // ASKED FOR, not decided here: the hierarchy comes from the coefficient
        // FIELDS, which only the matrix has (coefficients.hpp).
        std::shared_ptr<const gko::LinOp> precond = system.matrix().makePrecond(cfg_);
        if (cfg_.precondKind != la::PrecondKind::none && precond == nullptr)
        {
            throw std::runtime_error(
                "la::Solver: matrix format '" + std::string(system.matrix().name())
                + "' cannot build precond '" + cfg_.precond
                + "'; it builds its preconditioner from its own coefficients and declined this one"
            );
        }
        SystemKrylovSolver s(exec_, system, cfg_, std::move(precond));
        // KrylovSolver::solve takes a non-const rhs only because ISolver's
        // signature does; the rhs is READ (gather(const FA&), transfer.hpp) and
        // never written. The cast lives here, once, rather than widening either
        // interface.
        return s.solve(const_cast<amrex::MultiFab&>(system.rhs()), sol);
    }

private:

    // Runs a LinearSystem through the existing Krylov machinery: KrylovSolver
    // builds itself from a LinOp, and Matrix::op() is one.
    class SystemKrylovSolver : public la::KrylovSolver
    {
    public:

        SystemKrylovSolver(
            const NeoN::Executor& exec,
            const LinearSystem& system,
            const la::SolverConfig& cfg,
            std::shared_ptr<const gko::LinOp> precond
        )
            : la::KrylovSolver(
                la::makeExecutor(exec),
                // Global dimension, which every rank MUST agree on -- so it is
                // taken from the operator rather than recomputed here.
                system.matrix().op()->get_size()[0],
                system.localRows()
            )
        {
            // const_pointer_cast because Ginkgo's solver factories store a
            // non-const system matrix; nothing here writes through it.
            build(
                std::const_pointer_cast<gko::LinOp>(system.matrix().op()),
                cfg.solver,
                cfg.maxIter,
                cfg.rtol,
                cfg.atol,
                cfg.projectNullspace,
                std::move(precond),
                cfg.norm
            );
        }
    };

    NeoN::Executor exec_;
    la::SolverConfig cfg_;
};

} // namespace blockamr::la
