// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

// PRECONDITIONER CONSTRUCTION, in ONE place, for BOTH paths that need it.
//
// This is the seam that makes preconditioners reachable through la::Matrix. The
// three builders below used to live inside solve/persistent.cpp -- two of them
// inside FaceCoeffSolver's own constructor body -- which is why la::Solver could
// not reach them: the GMG hierarchy is built from the coefficient FIELDS, and by
// the time a solver holds an erased gko::LinOp the fields are gone.
//
// Moving them out does not change WHICH hierarchy is built. Every one of these is
// the pre-existing body, verbatim, with its parameters spelled out instead of
// captured; the callers below are unchanged in what they hand over and in what
// order. The bitwise GMG gate (plans/bench/compare_gmg_baseline.py) is the check
// on that claim.
//
// The DECLARATIONS are here, and only the declarations, so that
// faceCoeffMatrix.hpp can call them without pulling gmg/gmgPrecond.hpp,
// gmgKokkos/precond.hpp and AMReX_MLMG.H into every translation unit that names
// a matrix format. The definitions are in linearAlgebra/precond.cpp.

namespace blockamr
{
class KokkosGmgApply;
}

namespace blockamr::la
{

class GmgApplyMf;

// The hierarchy the native-GMG paths share: the stationary solver
// (GmgStationarySolver, persistent.cpp) drives it directly; the Krylov paths use
// it as an IR inner solver (solver="ir") or a preconditioner (precond="gmg").
struct GmgHierarchy
{
    std::shared_ptr<const gko::LinOp> op;
    const GmgApplyMf* mf = nullptr; // only read by the stationary path
};

GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    int precondCycles,
    const GmgConfig& gmg
);

/* @brief What a face-coefficient preconditioner build hands back.
 *
 * `op` is null for precond="none" (and for precond="mlmg" with no precond_mlmg
 * to wrap, which is the pre-existing "precond_mlmg alone implies mlmg" reading
 * run backwards). `kokkosVcycle` is set by precond="gmg_kokkos" ALONE: it is the
 * same V-cycle `op` wraps, held separately because solver="mpir" wraps it a
 * second time in an fp32 LinOp and building it twice would double the setup and
 * the device memory for two views of one cycle.
 */
struct FaceCoeffPrecond
{
    std::shared_ptr<const gko::LinOp> op;
    std::shared_ptr<blockamr::KokkosGmgApply> kokkosVcycle;
};

// Every preconditioner a MATRIX-FREE face-coefficient operator can carry:
// none / gmg / gmg_kokkos / mlmg. Throws for a combination this path genuinely
// cannot serve (gmg_kokkos with a Krylov bottom, an asymmetric cycle or a
// non-red-black smoother; mlmg without a precond_mlmg) -- those are refusals,
// not declines, and they carry FaceCoeffSolver's historical wording.
FaceCoeffPrecond makeFaceCoeffPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab* alpha,
    const amrex::MultiFab* ux,
    const amrex::MultiFab* lx,
    const amrex::MultiFab* uy,
    const amrex::MultiFab* ly,
    const amrex::MultiFab* uz,
    const amrex::MultiFab* lz,
    const amrex::Geometry& geom,
    const BcArray& bcArr,
    const SolverConfig& config
);

// An externally-built AMReX MLMG wrapped as a Ginkgo preconditioner. Null when
// the config carries no precond_mlmg -- there is nothing to wrap and that is not
// an error here (the "precond='mlmg' requires precond_mlmg" refusal belongs to
// the callers, which is where it has always been thrown).
std::shared_ptr<const gko::LinOp> makeMlmgPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab& alpha,
    const SolverConfig& config
);

} // namespace blockamr::la
