// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "NeoN/core/executor/executor.hpp"

#include <optional>
#include <string>

#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp" // FaceCoeffLevel

namespace blockamr::la
{

/* @brief What a ONE-SHOT solve takes beyond the system itself -- the union of the three
 *        `ginkgo_solve*` Python argument lists, so a binding lambda fills the fields its own
 *        surface exposes and leaves the rest at these defaults (which ARE the Python ones).
 *        Not a SolverConfig: nothing here is parsed to an enum, and `solver` must reach
 *        generateBasicSolver unvalidated so an unknown spelling still fails in its wording.
 *        Each entry point below names the fields it reads.
 */
struct OneshotSpec
{
    // "cg"/"bicgstab"/"gmres" (generateBasicSolver's subset).
    std::string solver = "bicgstab";
    int maxIter = 1000;
    // Quoted against the ORIGINAL system's ||rhs||, then applied absolutely; atol > 0 adds
    // the plain ||r_k||_2 <= atol.
    double rtol = 1e-10;
    double atol = 0.0;
    // sign*L must be SPD: -1 for MLPoisson, +1 for MLABecLaplacian. MLLinOp paths only.
    double sign = -1.0;
    // Where the Krylov vector ops run; resolved at CALL time. MLLinOp paths only.
    std::optional<NeoN::Executor> executor;
};

// Matrix-free residual-correction CG solve of the single-level MLLinOp system L(sol) = rhs;
// the body of `ginkgo_solve`, whose docstring carries the derivation and `sign`.
// Reads every OneshotSpec field but `solver`: this operator is SPD by construction and the
// method is always Cg.
SolveResult solveMlmgSystem(
    amrex::MLLinOpT<amrex::MultiFab>& lp,
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const OneshotSpec& spec
);

// Matrix-free solve of the multi-level COMPOSITE MLLinOp system, one sol/rhs MultiFab per
// AMR level (coarsest first); the body of `ginkgo_solve_composite`. Reads every field.
SolveResult solveComposite(
    amrex::MLLinOpT<amrex::MultiFab>& lp,
    const amrex::Vector<amrex::MultiFab*>& sol,
    const amrex::Vector<amrex::MultiFab const*>& rhs,
    const OneshotSpec& spec
);

// One-shot Ginkgo solve of a general face-coefficient system A(sol) = rhs; the body of
// `ginkgo_solve_face_coeffs`, whose historical dict has only num_iters/res_norm. The
// coefficients ARE the matrix, so `sign` is not read; nor is `executor` -- this entry point
// has no NeoN executor and its Ginkgo one is fixed to Reference.
SolveResult solveFaceCoeffs(
    const FaceCoeffLevel& level,
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const OneshotSpec& spec
);

} // namespace blockamr::la
