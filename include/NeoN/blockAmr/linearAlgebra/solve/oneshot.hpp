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

namespace blockamr::la
{

// Matrix-free residual-correction CG solve of the single-level MLLinOp system
// L(sol) = rhs; the body of the `ginkgo_solve` Python entry point, whose docstring
// (blockAmr/bindings/ginkgoSolve.cpp) carries the derivation and the `sign` contract.
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

// Matrix-free solve of the multi-level COMPOSITE MLLinOp system, one sol/rhs MultiFab
// per AMR level (coarsest first, unpacked from the Python lists by the caller); the
// body of `ginkgo_solve_composite` (docstring in bindings/ginkgoSolve.cpp).
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

// One-shot Ginkgo solve of a general structured face-coefficient system A(sol) = rhs;
// the body of `ginkgo_solve_face_coeffs` (docstring in bindings/ginkgoSolve.cpp).
// Historically returns only num_iters/res_norm, which the unset SolveResult fields
// preserve byte-for-byte at the nb::dict boundary.
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

} // namespace blockamr::la
