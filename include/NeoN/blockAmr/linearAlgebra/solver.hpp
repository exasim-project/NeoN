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
 * @brief Solves a LinearSystem, holding only an executor and a SolverConfig by value.
 *        Deliberately NO factory: parseSolverKind decided the method once. Returns
 *        la::SolveResult, as every solve entry point here does.
 */
class Solver
{
public:

    Solver(const NeoN::Executor& exec, const la::SolverConfig& cfg) : exec_(exec), cfg_(cfg) {}

    // rhs comes from the system, not the caller; `sol`'s values seed the initial guess.
    la::SolveResult solve(const LinearSystem& system, amrex::MultiFab& sol) const
    {
        // Refused: these three want the hierarchy as the SOLVER, a different object with a
        // different stopping test. Checked before the refusal pays for a hierarchy.
        if (cfg_.solverKind == la::SolverKind::gmg || cfg_.solverKind == la::SolverKind::ir
            || cfg_.solverKind == la::SolverKind::mpir)
        {
            throw std::runtime_error(
                "la::Solver: solver '" + cfg_.solver
                + "' needs the GMG hierarchy, which is built from the coefficient fields rather "
                  "than from a LinOp; use a Krylov method"
            );
        }
        // ASKED FOR, not decided here: the hierarchy comes from the coefficient FIELDS.
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
        // The rhs is READ, never written (gather takes a const FA&); the cast lives here,
        // once, rather than widening ISolver's signature.
        return s.solve(const_cast<amrex::MultiFab&>(system.rhs()), sol);
    }

private:

    // Runs a LinearSystem through the Krylov machinery: Matrix::op() is a LinOp.
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
                // Global dimension, which every rank MUST agree on -- so it comes from
                // the operator rather than being recomputed here.
                system.matrix().op()->get_size()[0],
                system.localRows()
            )
        {
            // const_pointer_cast: Ginkgo's factories store a non-const system matrix.
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
