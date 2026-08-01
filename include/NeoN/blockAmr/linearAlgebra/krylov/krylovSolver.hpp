// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <string>

#include "NeoN/blockAmr/core/gkoTypes.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/logging.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/result.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/stopNormInf.hpp"

namespace blockamr::la
{

// Anything FaceCoeffSolver can drive: pack rhs -> solve -> unpack sol. Defined here in full
// so the std::unique_ptr<ISolver> member needs no out-of-line destructor.
class ISolver
{
public:

    virtual ~ISolver() = default;

    virtual SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) = 0;
};

// Ginkgo Krylov solver built ONCE -- operator, generated solver and device scratch -- so a
// solve is pack -> apply -> unpack with no rebuild. The operator comes from a subclass.
class KrylovSolver : public ISolver
{
public:

    ~KrylovSolver() override = default;

    SolveResult solve(amrex::MultiFab& rhs, amrex::MultiFab& sol) override;

protected:

    // n is the GLOBAL cell count, nLocal this rank's share; equal outside MPI.
    KrylovSolver(
        std::shared_ptr<const gko::Executor> exec, gko::size_type n, gko::size_type nLocal
    );

    // Called by the subclass once its operator is built. `norm` selects what the criteria
    // and the reported res_norm measure in ("l2" | "linf", MLMG's; stopNormInf.hpp).
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

    // v -= mean(v), on the executor. Takes the GLOBAL view, not the local Dense, or each
    // rank would subtract its own partial mean.
    void subtractMean(gko::LinOp* v);

    std::shared_ptr<const gko::Executor> exec_;
    bool onDevice_;
    gko::size_type n_;
    gko::size_type nLocal_;
    std::shared_ptr<gko::LinOp> op_;
    // Rank-local storage, sized nLocal_. gather/scatter fill these directly.
    std::unique_ptr<Dense> b_;
    std::unique_ptr<Dense> x_;
    // What Ginkgo is handed: a distributed::Vector viewing the buffer above on >1 rank, the
    // buffer itself on one -- which is what makes its dots and norms reduce across ranks.
    std::shared_ptr<gko::LinOp> bGlobal_;
    std::shared_ptr<gko::LinOp> xGlobal_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> logger_;
    std::shared_ptr<ResidualHistoryLogger> resLogger_;
    bool projectNullspace_ = false;
    std::unique_ptr<Dense> ones_;
    std::shared_ptr<gko::LinOp> onesGlobal_;
    // Constant offset of an AFFINE operator (inhomogeneous BCs): op_ stays A and solve()
    // runs it on rhs - bcOffset_. Null on the homogeneous default.
    std::unique_ptr<Dense> bcOffset_;
    NormKind norm_ = NormKind::l2;
};

} // namespace blockamr::la
