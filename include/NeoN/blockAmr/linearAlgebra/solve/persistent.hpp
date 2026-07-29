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

#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/krylov.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"

namespace blockamr::la
{

// Anything the nanobind-visible facade (FaceCoeffSolver) can drive: pack rhs -> solve
// -> unpack sol. Implemented by the Ginkgo Krylov path (KrylovSolver below, also
// FaceCoeffCsrSolver's base) and by the native stationary V-cycle loop
// (GmgStationarySolver, file-local to persistent.cpp). Defined here in full so
// FaceCoeffSolver's std::unique_ptr<ISolver> member needs no out-of-line destructor.
class ISolver
{
public:

    virtual ~ISolver() = default;

    virtual SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) = 0;
};

// Ginkgo Krylov solver: operator, generated solver and device scratch vectors are built
// ONCE, so each solve is just pack rhs -> apply -> unpack sol with no per-call rebuild.
// The concrete operator comes from a subclass. Every instance unconditionally allocates
// its n-sized work vectors b_/x_; solver="gmg" never constructs one of these (it is a
// GmgStationarySolver, driving its own V-cycle on MultiFabs).
class KrylovSolver : public ISolver
{
public:

    ~KrylovSolver() override = default;

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

protected:

    // n is the GLOBAL cell count (the operators' dimension, which every rank must agree
    // on), nLocal this rank's share, which sizes the flat vectors; equal outside MPI.
    KrylovSolver(
        std::shared_ptr<const gko::Executor> exec, gko::size_type n, gko::size_type nLocal
    );

    // Called by the subclass once its operator is built. `norm` selects the norm the
    // stopping criteria -- and the reported res_norm -- measure in ("l2" | "linf",
    // MLMG's; see stopNormInf.hpp).
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

    // v -= mean(v), on the executor (dot with ones; only the scalar crosses to the
    // host). Uniform cells, so volume mean == arithmetic mean. Takes the GLOBAL view,
    // not the local Dense, or each rank would subtract its own partial mean.
    void subtractMean(gko::LinOp* v);

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    gko::size_type nLocal_;
    std::shared_ptr<gko::LinOp> op_;
    // Rank-local storage, sized nLocal_. gather/scatter fill these directly.
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    // What Ginkgo is handed: a distributed::Vector viewing the buffer above on >1 rank,
    // the buffer itself on one. The operators are elementwise and stay rank-local; what
    // these buy is that the dots and norms INSIDE the Krylov solver -- and in
    // subtractMean -- reduce across ranks, since its work vectors are clones of these.
    std::shared_ptr<gko::LinOp> bGlobal_;
    std::shared_ptr<gko::LinOp> xGlobal_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
    std::shared_ptr<ResidualHistoryLogger> resLogger_;
    bool projectNullspace_ = false;
    std::unique_ptr<Dense> ones_;
    std::shared_ptr<gko::LinOp> onesGlobal_;
    // Constant offset of an AFFINE operator (inhomogeneous domain BCs): op_ stays the
    // linear part A and solve() runs it on rhs - bcOffset_. Null on the default
    // homogeneous configuration, so nothing is allocated and nothing subtracted;
    // refreshed per solve by the subclass that owns the BC data.
    std::unique_ptr<Dense> bcOffset_;
    NormKind norm_ = NormKind::l2;
};

// Matrix-free persistent solver, and a pure facade: at construction it picks ONE
// strategy from config.solverKind (makeFaceCoeffSolver, persistent.cpp) -- the native
// stationary GMG V-cycle for solver="gmg", a Ginkgo Krylov solve (with the GMG
// hierarchy as optional preconditioner or IR inner solver) otherwise -- and forwards
// every solve() to it. Both strategies are file-local to persistent.cpp, so this class
// is the only part either Python or another translation unit ever names. Its operator
// references the caller's coefficient fields, so an in-place update can change the
// matrix with no reassembly -- see the staleness note in matrixFree/faceCoeffOp.hpp for
// when that does and does not hold.
class FaceCoeffSolver
{
public:

    // Non-const because the matrix-free operator takes the coefficients as mutable
    // CellFieldLevel/FaceFieldLevel handles (core/fieldLevel.hpp); nothing on this
    // path writes them.
    FaceCoeffSolver(
        const NeoN::Executor& executor,
        amrex::Geometry geom,
        amrex::MultiFab* alpha,
        amrex::MultiFab* ux,
        amrex::MultiFab* lx,
        amrex::MultiFab* uy,
        amrex::MultiFab* ly,
        amrex::MultiFab* uz,
        amrex::MultiFab* lz,
        const SolverConfig& config
    );

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol);

private:

    std::unique_ptr<ISolver> impl_;
};

// Assembled-CSR persistent solver: same matrix, stored explicitly -- its per-iteration
// SpMV streams the matrix from memory where FaceCoeffSolver recomputes entries from the
// face coefficients, which is the matrix-free comparison. Always a Ginkgo Krylov solve
// (there is no matrix-free CSR hierarchy), so it derives from KrylovSolver directly.
class FaceCoeffCsrSolver : public KrylovSolver
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
        const SolverConfig& config
    );
};

} // namespace blockamr::la
