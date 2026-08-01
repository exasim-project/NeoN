// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <string>
#include <vector>

#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylovSolver.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp" // FaceCoeffLevel
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

namespace blockamr::la
{

// Matrix-free persistent solver, and a pure facade: it picks ONE strategy from
// config.solverKind (makeFaceCoeffSolver) and forwards every solve() to it. Its operator
// references the caller's fields -- staleness rules in matrixFree/faceCoeffOp.hpp.
class FaceCoeffSolver
{
public:

    /* `level` carries MUTABLE field handles (the operator's staleness rules need them) plus
     * the ba/dm/geom they were allocated over -- the caller's fields, which must outlive this
     * solver. Nothing here writes them.
     */
    FaceCoeffSolver(
        const NeoN::Executor& executor, const FaceCoeffLevel& level, const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol);

private:

    std::unique_ptr<ISolver> impl_;
};

} // namespace blockamr::la
