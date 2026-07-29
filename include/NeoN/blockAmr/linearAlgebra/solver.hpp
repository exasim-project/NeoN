// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <stdexcept>
#include <string>

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
 * There is deliberately NO factory (design §7.4): a Solver is a value built from
 * a config, and which Krylov method that config names is decided inside, once,
 * where it already is (solverConfig.hpp's parseSolverKind).
 *
 * On the result type: this returns `la::SolveResult`, the struct every solve
 * entry point in this component already returns, rather than the design's
 * separate `SolverStats`. Its key set is a contract pinned by a test
 * (test_gmg_solver_stats_keys_match_cg), and a second nearly-identical struct is
 * the duplication this refactor exists to remove. See the S5 handoff.
 */
class Solver
{
public:

    Solver(const NeoN::Executor& exec, const la::SolverConfig& cfg) : exec_(exec), cfg_(cfg) {}

    // rhs comes from the system, not from the caller -- that is what carrying it
    // on LinearSystem buys (cf. NeoN::la::Solver::solve(ls, field)). `sol`'s
    // incoming values seed the initial guess, as everywhere else in this stack.
    la::SolveResult solve(const LinearSystem& system, amrex::MultiFab& sol) const
    {
        if (cfg_.precondKind != la::PrecondKind::none)
        {
            throw std::runtime_error(
                "la::Solver: precond '" + cfg_.precond
                + "' is not wired up yet (S5 drives the Krylov path only); use precond='none'"
            );
        }
        if (cfg_.solverKind == la::SolverKind::gmg || cfg_.solverKind == la::SolverKind::ir
            || cfg_.solverKind == la::SolverKind::mpir)
        {
            throw std::runtime_error(
                "la::Solver: solver '" + cfg_.solver
                + "' needs the GMG hierarchy, which is built from the coefficient fields rather "
                  "than from a LinOp; use a Krylov method"
            );
        }
        SystemKrylovSolver s(exec_, system, cfg_);
        // KrylovSolver::solve takes a non-const rhs only because ISolver's
        // signature does; the rhs is READ (gather(const FA&), transfer.hpp) and
        // never written. LinearSystem hands out a const rhs because a system is
        // data a solve consumes, so the cast is here, once, rather than widening
        // either interface.
        return s.solve(const_cast<amrex::MultiFab&>(system.rhs()), sol);
    }

private:

    // The one thing missing to run a LinearSystem through the existing Krylov
    // machinery: KrylovSolver builds itself from a LinOp, and Matrix::op() is one.
    class SystemKrylovSolver : public la::KrylovSolver
    {
    public:

        SystemKrylovSolver(
            const NeoN::Executor& exec, const LinearSystem& system, const la::SolverConfig& cfg
        )
            : la::KrylovSolver(
                la::makeExecutor(exec),
                // Global row/column dimension, which every rank must agree on:
                // taken from the operator itself, so nothing here has to
                // recompute a cell count the matrix already knows.
                system.matrix().op()->get_size()[0],
                system.localRows()
            )
        {
            // const_pointer_cast because build() hands the operator to Ginkgo's
            // solver factories, which store a non-const system matrix; op() is
            // const-correct on the format's side and nothing here writes through
            // it. Same cast, same reason, as the S4 test binding.
            build(
                std::const_pointer_cast<gko::LinOp>(system.matrix().op()),
                cfg.solver,
                cfg.maxIter,
                cfg.rtol,
                cfg.atol,
                cfg.projectNullspace,
                nullptr,
                cfg.norm
            );
        }
    };

    NeoN::Executor exec_;
    la::SolverConfig cfg_;
};

} // namespace blockamr::la
