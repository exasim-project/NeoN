// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <string>
#include <vector>

#include "../common/bc.hpp"
#include "../common/types.hpp"
#include "../gmg/gmg_precond.hpp"
#include "../krylov/executor.hpp"
#include "../krylov/krylov.hpp"
#include "../krylov/logging.hpp"
#include "../operators/face_coeff_op.hpp"

namespace nb = nanobind;

namespace blockamr::solvers
{

// Persistent solver: the operator, the generated Ginkgo solver and the device
// scratch vectors are built ONCE; each solve is just pack rhs -> apply ->
// unpack sol, reusing everything (no per-call operator/solver rebuild). The
// concrete operator is supplied by a subclass.
class PersistentSolver
{
public:

    virtual ~PersistentSolver() = default;

    virtual nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol);

protected:

    // allocDense=false skips the n-sized Ginkgo work vectors b_/x_ — the native
    // stationary solver (solver="gmg") drives the V-cycle on MultiFabs and never
    // touches them (a real memory saving at large N: 2 * n doubles).
    // n is the GLOBAL cell count -- the operators' row/column dimension, which
    // every rank must agree on. nLocal is the count this rank owns and is what
    // the flat vectors are sized by; they differ only under MPI.
    PersistentSolver(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type n,
        gko::size_type nLocal,
        bool allocDense = true
    );

    // Subclass calls this once its operator is built. `norm` selects the norm
    // the stopping criteria — and the reported res_norm — measure in ("l2" |
    // "linf", MLMG's; see stop_norm_inf.hpp).
    void build(
        std::shared_ptr<gko::LinOp> op,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        std::shared_ptr<const gko::LinOp> precond = nullptr,
        const std::string& norm = "l2"
    );

    // v -= mean(v), computed on the executor (dot with ones); only the scalar
    // mean crosses to the host. Uniform cells, so volume mean == arithmetic mean.
    // Takes the GLOBAL view (bGlobal_/xGlobal_), not the local Dense: the dot has
    // to reduce across ranks or each rank subtracts its own partial mean.
    void subtractMean(gko::LinOp* v);

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    gko::size_type nLocal_;
    std::shared_ptr<gko::LinOp> op_;
    // Rank-local storage, sized nLocal_. gather/scatter fill these directly.
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    // What Ginkgo is handed: a distributed::Vector viewing the buffer above on
    // >1 rank, the buffer itself on one. Everything the operators do to a vector
    // is elementwise and stays rank-local; what these buy is that the dots and
    // norms INSIDE the Krylov solver -- and in subtractMean below -- reduce
    // across ranks, because the solver's work vectors are clones of these.
    std::shared_ptr<gko::LinOp> bGlobal_;
    std::shared_ptr<gko::LinOp> xGlobal_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
    std::shared_ptr<ResidualHistoryLogger> resLogger_;
    bool projectNullspace_ = false;
    std::unique_ptr<Dense> ones_;
    std::shared_ptr<gko::LinOp> onesGlobal_;
    // Constant offset of an AFFINE operator (inhomogeneous domain BCs): op_ stays
    // the linear part A, and solve() runs it on rhs - bcOffset_. Null on every
    // homogeneous configuration, which is the default, so nothing is allocated
    // and nothing is subtracted unless a caller asked for inhomogeneous BCs.
    // Refreshed per solve by the subclass that owns the BC data.
    std::unique_ptr<Dense> bcOffset_;
    NormKind norm_ = NormKind::l2;
};

// Matrix-free persistent solver: the operator reads the caller's coefficient
// fields on the fly, so an external in-place update to them changes the matrix
// with no reassembly.
class FaceCoeffSolver : public PersistentSolver
{
public:

    FaceCoeffSolver(
        const NeoN::Executor& executor,
        amrex::Geometry geom,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        MLMG* precond_mlmg,
        int precond_cycles,
        const std::vector<std::string>& bc,
        const std::string& precond,
        int gmg_pre_sweeps,
        int gmg_post_sweeps,
        int gmg_coarsest_sweeps,
        int gmg_max_levels,
        int gmg_min_bottom,
        const std::string& gmg_smoother,
        const std::string& gmg_precision,
        const std::string& gmg_coeff_precision,
        double gmg_omega,
        int gmg_agg_l0_size,
        bool symmetric,
        const std::string& gmg_bottom_solver,
        int gmg_bottom_max_iter,
        double gmg_bottom_rtol,
        double mp_inner_rtol,
        int mp_inner_max_iter,
        const std::string& norm,
        const amrex::MultiFab* bc_data
    );

    // Native stationary GMG solver (solver="gmg") drives the V-cycle on MultiFabs;
    // every other solver keeps the base Krylov path. Dispatch here so the binding
    // (which calls S::solve on the concrete type) picks the right loop.
    nb::dict solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

private:

    // Build the precision-templated V-cycle hierarchy (fp64 default — byte-for-
    // byte the historical behaviour; fp32 halves the bandwidth-bound V-cycle
    // bytes, outer residual stays fp64). Also records the GmgApplyMf* so the
    // stationary solver can drive the V-cycle on fabs without knowing the type.
    std::shared_ptr<const gko::LinOp> buildGmgHierarchy(
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const amrex::Geometry& geom,
        const BcArray& bcArr,
        int precond_cycles,
        int gmg_pre_sweeps,
        int gmg_post_sweeps,
        int gmg_coarsest_sweeps,
        int gmg_max_levels,
        int gmg_min_bottom,
        const std::string& gmg_smoother,
        const std::string& gmg_precision,
        double gmg_omega,
        bool symmetric,
        const std::string& gmg_bottom_solver,
        int gmg_bottom_max_iter,
        double gmg_bottom_rtol
    );

    // Fill xWork_'s ghost layer for the FP64 residual: periodic/internal via
    // FillBoundary, then domain BCs via ghost reflection — the same fill
    // FaceCoeffOp does, so the residual uses the identical operator A.
    //
    // With bcData_ the reflection is the INHOMOGENEOUS one, which makes the outer
    // residual rhs - L(x) rather than rhs - A x. That is the whole of the
    // stationary path's inhomogeneous-BC support: the V-cycle then solves
    // A delta = rhs - L(x) with its own homogeneous fills, which is right because
    // a correction's boundary condition is homogeneous whatever the solution's
    // is, and the iteration converges to L(x) = rhs. No extra apply, no rhs fold
    // — the Krylov path needs both only because Ginkgo requires a linear operator.
    void fillGmgGhosts(amrex::MultiFab& mf) const;

    // mf -= mean(mf) over the valid region (constant-nullspace projection for
    // singular systems; uniform cells so the volume mean is the arithmetic mean).
    void subtractMeanMf(amrex::MultiFab& mf) const;

    // Native stationary V-cycle solve: x <- x + V(b - A x), warm-started from the
    // incoming sol, until ||r|| <= max(rtol*||b||, atol) or max_iter cycles. Runs
    // entirely on AMReX fabs — no Ginkgo Krylov object, no per-iteration
    // flat-vector pack/unpack, no per-iteration Ginkgo<->AMReX crossings.
    nb::dict gmgSolve(amrex::MultiFab& rhs, amrex::MultiFab& sol);

    // Native stationary GMG solver state (only populated when solver="gmg").
    bool gmgStationary_ = false;
    const amrex::MultiFab* alpha_ = nullptr;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    amrex::Geometry geom_ {};
    BcArray bcArr_ {};
    bool hasPhysBc_ = false;
    // Inhomogeneous domain-BC data, null when the BCs are homogeneous. bcData_
    // drives the stationary path's residual fill (pinned in ownedBcData_ on the
    // host); bcOffsetOp_ is the typed hook into op_ the Krylov path calls once
    // per solve to refresh PersistentSolver::bcOffset_.
    const amrex::MultiFab* bcData_ = nullptr;
    std::shared_ptr<amrex::MultiFab> ownedBcData_;
    const FaceCoeffOp* bcOffsetOp_ = nullptr;
    int maxIter_ = 0;
    double rtol_ = 0.0;
    double atol_ = 0.0;
    bool projectNull_ = false;
    std::shared_ptr<const gko::LinOp> gmgOwner_;               // keeps the V-cycle hierarchy alive
    const GmgApplyMf* gmgMf_ = nullptr;                        // typed V-cycle hook into gmgOwner_
    std::shared_ptr<amrex::MultiFab> xWork_;                   // FP64 iterate (1 ghost)
    std::shared_ptr<amrex::MultiFab> rhsPinned_;               // pinned rhs stage (reference path)
    std::vector<std::shared_ptr<amrex::MultiFab>> ownedCoeff_; // pinned coeffs (reference path)
};

// Assembled-CSR persistent solver: same matrix, stored explicitly. Its per-
// iteration SpMV streams the matrix from memory, versus FaceCoeffSolver which
// recomputes entries from the face coefficients — the matrix-free comparison.
class FaceCoeffCsrSolver : public PersistentSolver
{
public:

    FaceCoeffCsrSolver(
        const NeoN::Executor& executor,
        amrex::Geometry geom,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        MLMG* precond_mlmg,
        int precond_cycles,
        const std::vector<std::string>& bc,
        const std::string& precond,
        int /*gmg_pre_sweeps*/,
        int /*gmg_post_sweeps*/,
        int /*gmg_coarsest_sweeps*/,
        int /*gmg_max_levels*/,
        int /*gmg_min_bottom*/,
        const std::string& /*gmg_smoother*/,
        const std::string& /*gmg_precision*/,
        const std::string& /*gmg_coeff_precision*/,
        double /*gmg_omega*/,
        int /*gmg_agg_l0_size*/,
        bool /*symmetric*/,
        const std::string& /*gmg_bottom_solver*/,
        int /*gmg_bottom_max_iter*/,
        double /*gmg_bottom_rtol*/,
        double /*mp_inner_rtol*/,
        int /*mp_inner_max_iter*/,
        const std::string& norm,
        const amrex::MultiFab* bc_data
    );
};

} // namespace blockamr::solvers
