// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

// Preconditioner construction in ONE place -- the seam that makes them reachable from a
// la::MFFaceCoeffs (through la::makeHierarchy, ginkgo/adapt.hpp): the GMG hierarchy comes
// from the coefficient FIELDS, not from a LinOp. DECLARATIONS ONLY (definitions in
// precond.cpp), and bitwise equivalence to the pre-existing hierarchies is gated by
// plans/bench/compare_gmg_baseline.py.

namespace blockamr
{
class KokkosGmgApply;
}

namespace blockamr::la
{

class GmgApplyMf;

// Shared by the native-GMG paths: driven directly by GmgStationarySolver, or used as an
// IR inner solver (solver="ir") / preconditioner (precond="gmg").
struct GmgHierarchy
{
    std::shared_ptr<const gko::LinOp> op;
    const GmgApplyMf* mf = nullptr; // only read by the stationary path
};

// The cycle count and the hierarchy knobs come from the config, not loose: every caller
// already holds the SolverConfig those two live in.
GmgHierarchy buildGmgHierarchy(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
);

/* @brief What a face-coefficient preconditioner build hands back. `op` is null for
 *        precond="none" and for "mlmg" without a precond_mlmg. `kokkosVcycle` is the same
 *        V-cycle `op` wraps, held separately because solver="mpir" wraps it again.
 */
struct FaceCoeffPrecond
{
    std::shared_ptr<const gko::LinOp> op;
    std::shared_ptr<blockamr::KokkosGmgApply> kokkosVcycle;
};

// Every preconditioner a MATRIX-FREE face-coefficient operator can carry: none / gmg /
// gmg_kokkos / mlmg. THROWS (a refusal, in FaceCoeffSolver's wording) for a combination
// this path cannot serve: gmg_kokkos with a Krylov bottom, asymmetric or non-red-black.
FaceCoeffPrecond makeFaceCoeffPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const FaceCoeffLevel& level,
    const BcArray& bcArr,
    const SolverConfig& config
);

// An externally-built AMReX MLMG wrapped as a Ginkgo preconditioner. Null, not an error,
// when the config carries no precond_mlmg: that refusal belongs to the callers.
std::shared_ptr<const gko::LinOp> makeMlmgPrecond(
    std::shared_ptr<const gko::Executor> exec,
    gko::size_type n,
    const amrex::MultiFab& alpha,
    const SolverConfig& config
);

} // namespace blockamr::la
