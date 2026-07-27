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

#include "../krylov/result.hpp"

namespace blockamr::solvers
{

// Matrix-free residual-correction CG solve of the single-level MLLinOp system
// L(sol) = rhs. The body of the `ginkgo_solve` Python entry point; see its
// docstring in bindings/blockAMR/ginkgo_solve.cpp for the full
// residual-correction derivation and the `sign` contract.
SolveResult solveMlmgSystem(
    amrex::MLLinOpT<amrex::MultiFab>& lp,
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    int max_iter,
    double rtol,
    double atol,
    double sign,
    std::optional<NeoN::Executor> executor
);

// Matrix-free solve of the multi-level COMPOSITE MLLinOp system, one sol/rhs
// MultiFab per AMR level (coarsest first, already unpacked from the Python
// lists by the caller). The body of the `ginkgo_solve_composite` Python entry
// point; see its docstring in bindings/blockAMR/ginkgo_solve.cpp.
SolveResult solveComposite(
    amrex::MLLinOpT<amrex::MultiFab>& lp,
    const amrex::Vector<amrex::MultiFab*>& sol,
    const amrex::Vector<amrex::MultiFab const*>& rhs,
    int max_iter,
    double rtol,
    double atol,
    double sign,
    std::optional<NeoN::Executor> executor,
    const std::string& solver
);

// One-shot Ginkgo solve of a general structured face-coefficient system
// A(sol) = rhs. The body of the `ginkgo_solve_face_coeffs` Python entry point;
// see its docstring in bindings/blockAMR/ginkgo_solve.cpp. Historically
// returns only num_iters/res_norm (no converged/res_history/contraction/
// diagnostic); the unset SolveResult fields preserve that byte-for-byte at
// the nb::dict boundary.
SolveResult solveFaceCoeffs(
    amrex::MultiFab& alpha,
    amrex::MultiFab& ux,
    amrex::MultiFab& lx,
    amrex::MultiFab& uy,
    amrex::MultiFab& ly,
    amrex::MultiFab& uz,
    amrex::MultiFab& lz,
    amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::Geometry& geom,
    const std::string& solver,
    int max_iter,
    double rtol
);

} // namespace blockamr::solvers
