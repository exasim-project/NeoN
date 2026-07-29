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

// Anything the nanobind-visible solver facade (FaceCoeffSolver) can drive:
// pack rhs -> solve -> unpack sol. Two things implement it, both file-local
// to persistent.cpp: the Ginkgo Krylov path (KrylovSolver below -- also
// FaceCoeffCsrSolver's own base) and the native stationary V-cycle loop
// (GmgStationarySolver). FaceCoeffSolver picks one at construction time (see
// makeFaceCoeffSolver in persistent.cpp) and forwards every solve() call to
// it. Defined here in full (not forward-declared) so FaceCoeffSolver's
// std::unique_ptr<ISolver> member never needs an out-of-line destructor.
class ISolver
{
public:

    virtual ~ISolver() = default;

    virtual SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) = 0;
};

// Ginkgo Krylov solver: the operator, the generated Ginkgo solver and the
// device scratch vectors are built ONCE; each solve is just pack rhs -> apply
// -> unpack sol, reusing everything (no per-call operator/solver rebuild).
// The concrete operator is supplied by a subclass. Every KrylovSolver
// unconditionally allocates its n-sized Ginkgo work vectors (b_/x_) --  the
// native stationary solver (solver="gmg") never constructs one of these at
// all; it is a GmgStationarySolver instead, which drives its own V-cycle on
// MultiFabs and needs neither.
class KrylovSolver : public ISolver
{
public:

    ~KrylovSolver() override = default;

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

protected:

    // n is the GLOBAL cell count -- the operators' row/column dimension, which
    // every rank must agree on. nLocal is the count this rank owns and is what
    // the flat vectors are sized by; they differ only under MPI.
    KrylovSolver(
        std::shared_ptr<const gko::Executor> exec, gko::size_type n, gko::size_type nLocal
    );

    // Subclass calls this once its operator is built. `norm` selects the norm
    // the stopping criteria -- and the reported res_norm -- measure in ("l2" |
    // "linf", MLMG's; see stopNormInf.hpp).
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
// fields on the fly, so an external in-place update to them changes the
// matrix with no reassembly. A pure facade: at construction it picks ONE of
// two solve strategies from config.solverKind (see makeFaceCoeffSolver,
// persistent.cpp) -- the native stationary GMG V-cycle for solver="gmg", a
// Ginkgo Krylov solve (with the GMG hierarchy as an optional preconditioner
// or IR inner solver) for everything else -- and forwards every solve() call
// to whichever it built. Both strategies are file-local to persistent.cpp
// (GmgStationarySolver, FaceCoeffKrylovSolver); this class is the only part
// of either that Python, or any other translation unit, ever names.
class FaceCoeffSolver
{
public:

    // The coefficients are non-const because the matrix-free operator takes them
    // as CellFieldLevel/FaceFieldLevel handles (core/fieldLevel.hpp), which are
    // mutable handles; the const on these parameters was a declaration, not a
    // property of the caller -- the binding holds them as amrex::MultiFab&.
    // Nothing on this path writes them.
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

// Assembled-CSR persistent solver: same matrix, stored explicitly. Its per-
// iteration SpMV streams the matrix from memory, versus FaceCoeffSolver which
// recomputes entries from the face coefficients -- the matrix-free comparison.
// Always a Ginkgo Krylov solve (there is no matrix-free CSR hierarchy), so
// this derives from KrylovSolver directly -- no facade, no extra members, no
// overrides.
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
