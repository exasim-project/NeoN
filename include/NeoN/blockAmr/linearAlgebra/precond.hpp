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

// Preconditioner construction in ONE place, for BOTH paths that need it -- the
// seam that makes preconditioners reachable through la::Matrix, which cannot
// build them itself because the GMG hierarchy comes from the coefficient FIELDS
// and an erased gko::LinOp no longer has them.
//
// Bitwise-equivalence to the pre-existing hierarchies is gated by
// plans/bench/compare_gmg_baseline.py.
//
// DECLARATIONS ONLY, so faceCoeffMatrix.hpp can call these without pulling
// gmg/gmgPrecond.hpp, gmgKokkos/precond.hpp and AMReX_MLMG.H into every TU that
// names a matrix format. Definitions in linearAlgebra/precond.cpp.

namespace blockamr
{
class KokkosGmgApply;
}

namespace blockamr::la
{

class GmgApplyMf;

// Shared by the native-GMG paths: driven directly by GmgStationarySolver
// (persistent.cpp), or used as an IR inner solver (solver="ir") / preconditioner
// (precond="gmg").
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
 * `op` is null for precond="none", and for precond="mlmg" with no precond_mlmg to
 * wrap. `kokkosVcycle` is set by precond="gmg_kokkos" ALONE and is the same
 * V-cycle `op` wraps, held separately because solver="mpir" wraps it again in an
 * fp32 LinOp and building it twice would double setup time and device memory.
 */
struct FaceCoeffPrecond
{
    std::shared_ptr<const gko::LinOp> op;
    std::shared_ptr<blockamr::KokkosGmgApply> kokkosVcycle;
};

// Every preconditioner a MATRIX-FREE face-coefficient operator can carry:
// none / gmg / gmg_kokkos / mlmg. THROWS (a refusal, not a decline, in
// FaceCoeffSolver's historical wording) for a combination this path cannot serve:
// gmg_kokkos with a Krylov bottom, an asymmetric cycle or a non-red-black
// smoother; mlmg without a precond_mlmg.
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

// An externally-built AMReX MLMG wrapped as a Ginkgo preconditioner. Null, not an
// error, when the config carries no precond_mlmg: the "precond='mlmg' requires
// precond_mlmg" refusal belongs to the callers and is thrown there.
std::shared_ptr<const gko::LinOp> makeMlmgPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab& alpha,
    const SolverConfig& config
);

} // namespace blockamr::la
