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

#include "bc.hpp"
#include "gmg_precond.hpp"
#include "krylov.hpp"
#include "types.hpp"

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
    PersistentSolver(
        std::shared_ptr<const gko::Executor> exec, gko::size_type n, bool allocDense = true
    );

    // Subclass calls this once its operator is built.
    void build(
        std::shared_ptr<gko::LinOp> op,
        const std::string& solver,
        int max_iter,
        double rtol,
        double atol,
        bool project_nullspace,
        std::shared_ptr<const gko::LinOp> precond = nullptr
    );

    // v -= mean(v), computed on the executor (dot with ones); only the scalar
    // mean crosses to the host. Uniform cells, so volume mean == arithmetic mean.
    void subtractMean(Dense* v);

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    std::shared_ptr<gko::LinOp> op_;
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
    std::shared_ptr<ResidualHistoryLogger> resLogger_;
    bool projectNullspace_ = false;
    std::unique_ptr<Dense> ones_;
};

// Matrix-free persistent solver: the operator reads the caller's coefficient
// fields on the fly, so an external in-place update to them changes the matrix
// with no reassembly.
class FaceCoeffSolver : public PersistentSolver
{
public:

    FaceCoeffSolver(
        const std::string& executor,
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
        double gmg_omega
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
        double gmg_omega
    );

    // Fill xWork_'s ghost layer for the FP64 residual: periodic/internal via
    // FillBoundary, then homogeneous domain BCs via ghost reflection — the same
    // fill FaceCoeffOp does, so the residual uses the identical operator A.
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
        const std::string& executor,
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
        double /*gmg_omega*/
    );
};

} // namespace blockamr::solvers
